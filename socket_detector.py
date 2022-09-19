import socket
import numpy as np
import cv2
from classifer import WoodClass
import time
import os

from root_dir import ROOT_DIR
from utils import PreSocket, receive_sock, parse_protocol, ack_sock, done_sock
import logging


def try_connect(is_repeat: bool = False, max_reconnect_times: int = 50) -> (bool, socket.socket):
    """
    尝试连接.

    :param is_repeat: 是否是重新连接
    :param max_reconnect_times:最大重连次数
    :return: (连接状态True为成功, Socket / None)
    """
    reconnect_time = 0
    while reconnect_time < max_reconnect_times:
        logging.warning(f'尝试{"重新" if is_repeat else ""}发起第{reconnect_time+1}次连接...')
        try:
            connected_sock = PreSocket(socket.AF_INET, socket.SOCK_STREAM)
            connected_sock.connect(('127.0.0.1', 23456))
        except Exception as e:
            reconnect_time += 1
            logging.error(f'第{reconnect_time}次连接失败\n {e}')
            continue
        logging.warning(f'{"重新" if is_repeat else ""}连接成功')
        return True, connected_sock
    return False, None


def process_cmd(cmd: str, data: any, connected_sock: PreSocket, detector: WoodClass) -> bool:
    """
    处理指令

    :param cmd: 指令类型
    :param data: 指令内容
    :param connected_sock: socket
    :param detector: 模型
    :return: 是否处理成功
    """

    if cmd == 'IM':
        wood_color = detector.predict(data)
        response = done_sock(connected_sock, cmd_type=cmd, result=wood_color)
    elif cmd == 'TR':
        detector = WoodClass(w=4096, h=1200, n=3000, debug_mode=False)
        detector.fit_pictures(data_path=data)
        response = done_sock(connected_sock, cmd_type=cmd)
    elif cmd == 'MD':
        detector.load(path=data)
        response = done_sock(connected_sock, cmd_type=cmd)
    else:
        logging.error(f'错误指令，指令为{cmd}')
        response = False
    return response


def main():
    status, connected_sock = False, None
    while not status:
        status, connected_sock = try_connect()
    # socket_send = PreSocket(socket.AF_INET, socket.SOCK_STREAM)
    # socket_send.connect(('127.0.0.1', 21123))
    # a = b'\xaa\x00\x00\x00\x05\x20\x44\x54\x52\xff\xff\xff\xbb'
    # connected_sock.send(a)
    model_path = os.path.join(ROOT_DIR, "models/model_2022-09-06_13-08.p")
    detector = WoodClass(w=4096, h=1200, n=3000, debug_mode=False)
    detector.load(path=model_path)
    while True:
        pack, next_pack = receive_sock(connected_sock)
        if pack == b"":
            status, connected_sock = try_connect()
            continue
        connected_sock.set_prepack(next_pack)

        cmd, data = parse_protocol(pack)
        ack_sock(connected_sock, cmd_type=cmd)
        process_cmd(cmd=cmd, data=data, connected_sock=connected_sock, detector=detector)


if __name__ == '__main__':
    # 2个端口
    # 接受端口21122
    # 发送端口21123
    # 接收到图片 n_rows * n_bands * n_cols, float32
    # 发送图片 n_rows * n_cols, uint8
    main()
    # test(r"D:\build-tobacco-Desktop_Qt_5_9_0_MSVC2015_64bit-Release\calibrated15.raw")
    # main()
    # debug_main()
    # test_run(all_data_dir=r'D:\数据')
    # with open(r'D:\数据\虫子\valid2.raw', 'rb') as f:
    #     data = np.frombuffer(f.read(), dtype=np.float32).reshape(600, 29, 1024).transpose(0, 2, 1)
    # plt.matshow(data[:, :, 10])
    # plt.show()
    # detector = SpecDetector('model_spec/model_29.p')
    # result = detector.predict(data)
    #
    # plt.matshow(result)
    # plt.show()
    # result = result.reshape((600, 1024))
