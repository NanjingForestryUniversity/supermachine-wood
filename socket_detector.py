import socket
import numpy as np
import cv2
from classifer import WoodClass
import time
import os
from utils import PreSocket, receive_sock, parse_protocol, ack_sock, done_sock
import logging


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
            wood_color = detector.predict(data)
            done_sock(send_sock, cmd_type=cmd, result=wood_color)
        elif cmd == 'TR':
            detector.fit_pictures(data_path=r"C:\Users\FEIJINTI\PycharmProjects\wood_color")
            done_sock(send_sock, cmd_type=cmd)
        elif cmd == 'MD':
            model_path = os.path.join("models", data)
            detector.load(path=model_path)
            done_sock(send_sock, cmd_type=cmd)
            print(model_path)
        else:
            logging.error(f'错误指令，指令为{cmd}')


def main():
    socket_receive = PreSocket(socket.AF_INET, socket.SOCK_STREAM)
    socket_receive.connect(('127.0.0.1', 23456))
    # socket_send = PreSocket(socket.AF_INET, socket.SOCK_STREAM)
    # socket_send.connect(('127.0.0.1', 21123))
    process_cmd(recv_sock=socket_receive, send_sock=socket_receive)


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
