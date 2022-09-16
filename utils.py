# -*- coding: utf-8 -*-
"""
Created on Nov 3 21:18:26 2020

@author: l.z.y
@e-mail: li.zhenye@qq.com
"""
import logging
import os
import shutil
import time
import socket
from classifer import WoodClass


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def create_file(file_name):
    """
    创建文件
    :param file_name: 文件名
    :return: None
    """
    if os.path.exists(file_name):
        print("文件存在：%s" % file_name)
        return False
        # os.remove(file_name)  # 删除已有文件
    if not os.path.exists(file_name):
        print("文件不存在，创建文件：%s" % file_name)
        open(file_name, 'a').close()
        return True


class Logger(object):
    def __init__(self, is_to_file=False, path=None):
        self.is_to_file = is_to_file
        if path is None:
            path = "wood.log"
        self.path = path
        create_file(path)

    def log(self, content):
        if self.is_to_file:
            with open(self.path, "a") as f:
                print(time.strftime("[%Y-%m-%d_%H-%M-%S]:"), file=f)
                print(content, file=f)
        else:
            print(content)


class PreSocket(socket.socket):
    def __int__(self, *args, **kwargs):
        super(socket.socket, self).__init__(*args, **kwargs)
        self.pre_pack = b''

    def receive(self, *args, **kwargs):
        if len(self.pre_pack) == 0:
            self.recv(*args, **kwargs)
        else:
            data_len = args[0]
            required, left = self.pre_pack[:data_len], self.pre_pack[data_len:]
            self.pre_pack = left
            return required

    def set_prepack(self, pre_pack: bytes):
        self.pre_pack += pre_pack


def process_cmd(recv_sock: PreSocket, send_sock: PreSocket):
    model_path = "models/model_2022-09-06_13-08.p"
    detector = WoodClass(w=4096, h=1200, n=3000, debug_mode=False)
    detector.load(path=model_path)
    while True:
        pack, next_pack = receive_sock(recv_sock)
        recv_sock.set_prepack(next_pack)
        cmd, data = parse_protocol(pack)
        ack_sock(send_sock, cmd_type=cmd)
        if cmd == 'IM':
            wood_color = str(detector.predict(data))
            done_sock(send_sock, cmd_type=cmd, result=wood_color)
        elif cmd == 'TR':
            detector.fit_pictures(data_path=r"C:\Users\FEIJINTI\PycharmProjects\wood_color")
            done_sock(send_sock, cmd_type=cmd)
        elif cmd == 'MD':
            model_path = os.path.join("models", data)
            detector.load(path=model_path)
            done_sock(send_sock, cmd_type=cmd)
        else:
            logging.error(f'错误指令，指令为{cmd}')


def receive_sock(recv_sock: PreSocket, pre_pack: bytes = b'') -> (bytes, bytes):
    """
    从指定的socket中读取数据.

    :param recv_sock: 指定sock
    :param pre_pack: 上一包的粘包内容
    :return: data, next_pack
    """
    recv_sock.set_prepack(pre_pack)
    # 开头校验
    while True:
        try:
            temp = recv_sock.receive(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return '', None
        except TimeoutError as e:
            logging.error(f'超时了，错误代码: \n{e}')
            return '', None
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return '', None
        if temp == b'aa':
            break

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文长度失败, 错误代码: \n{e}')
            return b'', b''
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}, \n报文内容\n{temp}')
        return b'', b''

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        try:
            temp += recv_sock.receive(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}，\n报文内容\n{temp}')
            return b'', b''
    data, next_pack = temp[:data_len], temp[data_len:]

    # 进行数据校验
    temp = b''
    while len(temp) < 3:
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文校验失败, 错误代码: \n{e}')
            return b'', b''
    if temp == b'\xff\xff\xbb':
        return data, next_pack
    else:
        logging.error(f"接收了一个完美的只错了校验位的报文，\n data: {data} \n next_pack:{next_pack}")
        return b'', b''


def parse_protocol(data: bytes) -> (str, any):
    '''
    指令转换.

    :param data:接收到的报文
    :return: 指令类型和内容
    '''
    try:
        assert len(data) > 4
    except AssertionError:
        logging.error('指令转换失败，长度不足5')
        return '', None
    cmd, data = data[:4], data[4:]
    cmd = cmd.decode('ascii').strip().upper()
    if cmd == 'IM':
        n_rows, n_cols, img = data[:2], data[2:4], data[4:]
        try:
            n_rows, n_cols = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文内容: n_rows:{n_rows}, n_cols: {n_cols}')
            return '', None
        try:
            assert n_rows * n_cols * 3 == len(img)
        except AssertionError:
            logging.error('图像指令IM转换失败，数据长度错误')
            return '', None
        img = np.frombuffer(img, dtype=np.uint8).reshape((n_rows, n_cols, -1))
        return cmd, img
    elif cmd == 'TR':
        return cmd, None
    elif cmd == 'MD':
        data = data.decode('ascii')
        return cmd, data


def ack_sock(send_sock:PreSocket, cmd_type: str) -> bool:
    '''
    发送应答
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :return:是否发送成功
    '''
    msg = b'\xaa\x00\x00\x00\xd8'+(' A'+cmd_type).upper().encode('ascii')+b'\xff\xff\xff\xbb'
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送应答失败，错误类型：{e}')
        return False
    return True


def done_sock(send_sock: PreSocket, cmd_type: str, result: str = '') -> bool:
    '''
    发送任务完成指令
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :param result:数据
    :return:是否发送成功
    '''
    cmd_type = cmd_type.strip().upper()
    if (cmd_type == "TR") or (cmd_type == "MD"):
        if result != '':
            logging.error('结果在这种指令里很没必要')
        result = b'\xff'
    elif cmd_type == 'IM':
        result = result.encode('ascii')
    msg = b'\xaa\x00\x00\x00\xd8'+(' D'+cmd_type).upper().encode('ascii') + result + b'\xff\xff\xbb'
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False
    return True


if __name__ == '__main__':
    log = Logger(is_to_file=True)
    log.log("nihao")
    import numpy as np
    a = np.ones((100, 100, 3))
    log.log(a.shape)
