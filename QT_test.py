# -*- codeing = utf-8 -*-
# Time : 2022/9/17 15:05
# @Auther : zhouchao
# @File: QT_test.py
# @Software:PyCharm
import logging
import socket
import numpy as np
import cv2
import binascii


def rec_socket(recv_sock: socket.socket, cmd_type: str, ack: bool) -> bool:
    if ack:
        cmd = 'A' + cmd_type
    else:
        cmd = 'D' + cmd_type
    while True:
        try:
            temp = recv_sock.recv(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return False
        except TimeoutError as e:
            logging.error(f'超时了，错误代码: \n{e}')
            return False
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return False
        if temp == b'\xaa':
            break

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        try:
            temp += recv_sock.recv(1)
        except Exception as e:
            logging.error(f'接收报文长度失败, 错误代码: \n{e}')
            return False
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}, \n报文内容\n{temp}')
        return False

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        try:
            temp += recv_sock.recv(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}，\n报文内容\n{temp}')
            return False
    data = temp
    if cmd.strip().upper() != data[:4].decode('ascii').strip().upper():
        logging.error(f'客户端接收指令错误,\n指令内容\n{data}')
        return False
    else:
        if cmd == 'DIM':
            print(data)

        # 进行数据校验
        temp = b''
        while len(temp) < 3:
            try:
                temp += recv_sock.recv(1)
            except Exception as e:
                logging.error(f'接收报文校验失败, 错误代码: \n{e}')
                return False
        if temp == b'\xff\xff\xbb':
            return True
        else:
            logging.error(f"接收了一个完美的只错了校验位的报文，\n data: {data}")
            return False


def main():
    socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_receive.bind(('127.0.0.1', 23456))
    socket_receive.listen(5)
    # socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # socket_send.bind(('127.0.0.1', 21122))
    # socket_send.listen(5)
    print('等待连接')
    socket_send_1, receive_addr_1 = socket_receive.accept()
    print("连接成功：", receive_addr_1)
    socket_send_2 = socket_send_1
    # socket_send_2, receive_addr_2 = socket_receive.accept()
    # print("连接成功：", receive_addr_2)
    while True:
        cmd = input().strip().upper()
        if cmd == 'IM':
            img = cv2.imread(r"C:\Users\FEIJINTI\PycharmProjects\wood_color\data\dark\rgb60.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width = img.shape[0]
            height = img.shape[1]
            length = width * height * 3 + 4 + 4
            length = length.to_bytes(4, byteorder='big')
            width = width.to_bytes(2, byteorder='big')
            height = height.to_bytes(2, byteorder='big')
            send_message = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + width + height
            socket_send_1.send(send_message)
            socket_send_1.send(img)
            socket_send_1.send(b'\xff\xff\xbb')
            print('发送成功')
            if rec_socket(socket_send_2, cmd_type=cmd, ack=True):
                print('接收指令成功')
            else:
                print('接收指令失败')
            if rec_socket(socket_send_2, cmd_type=cmd, ack=False):
                print('指令执行完毕')
            else:
                print('指令执行失败')
        elif cmd == 'TR':
            send_message = b'\xaa\x00\x00\x00\x05' + ('  ' + cmd).upper().encode('ascii') + b'\xff\xff\xff\xbb'
            socket_send_1.send(send_message)
            print('发送成功')
            if rec_socket(socket_send_2, cmd_type=cmd, ack=True):
                print('接收指令成功')
            else:
                print('接收指令失败')
            if rec_socket(socket_send_2, cmd_type=cmd, ack=False):
                print('指令执行完毕')
            else:
                print('指令执行失败')
        elif cmd == 'MD':
            model = "model_2020-11-08_20-49.p"
            model = model.encode('ascii')
            length = len(model) + 4
            length = length.to_bytes(4, byteorder='big')
            send_message = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + model + b'\xff\xff\xbb'
            socket_send_1.send(send_message)
            print('发送成功')
            if rec_socket(socket_send_2, cmd_type=cmd, ack=True):
                print('接收指令成功')
            else:
                print('接收指令失败')
            if rec_socket(socket_send_2, cmd_type=cmd, ack=False):
                print('指令执行完毕')
            else:
                print('指令执行失败')
        else:
            print('指令错误')


if __name__ == '__main__':
    main()

