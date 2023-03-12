import socket
import sys

import numpy as np
import cv2

import root_dir
from classifer import WoodClass
import time
import os

from database import Database
from root_dir import ROOT_DIR
from utils import PreSocket, receive_sock, parse_protocol, ack_sock, done_sock, DualSock, simple_sock
import logging
from config import Config


def process_cmd(cmd: str, data: any, connected_sock: socket.socket, detector: WoodClass, settings: Config) -> tuple:
    """
    处理指令

    :param cmd: 指令类型
    :param data: 指令内容
    :param connected_sock: socket
    :param detector: 模型
    :return: 是否处理成功
    """
    result = ''
    if cmd == 'IM':
        wood_color = detector.predict(data)
        result = {0: 'dark', 1: 'middle', 2: 'light'}[wood_color]
        response = simple_sock(connected_sock, cmd_type=cmd, result=wood_color)
    elif cmd == 'TR':
        detector = WoodClass(w=4096, h=1200, n=3000, debug_mode=False)
        model_name = None
        if ":" in data:
            data, model_name = data.split(":", 1)
            model_name = model_name + ".p"
        settings.data_path = data
        settings.model_path = ROOT_DIR / 'models' / detector.fit_pictures(data_path=settings.data_path, file_name=model_name)
        response = simple_sock(connected_sock, cmd_type=cmd)
    elif cmd == 'MD':
        settings.model_path = data
        detector.load(path=settings.model_path)
        response = simple_sock(connected_sock, cmd_type=cmd)
    else:
        logging.error(f'错误指令，指令为{cmd}')
        response = False
    return response, result


def main(is_debug=False):
    settings = Config()
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'report.log'))
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler], level=logging.DEBUG)
    dual_sock = DualSock(connect_ip='127.0.0.1')

    database = Database(settings.database_addr)


    while not dual_sock.status:
        dual_sock.reconnect()
    detector = WoodClass(w=4096, h=1200, n=3000, debug_mode=False)
    detector.load(path=settings.model_path)
    _ = detector.predict(np.random.randint(1, 254, (1200, 4096, 3), dtype=np.uint8))
    while True:
        pack, next_pack = receive_sock(dual_sock)
        if pack == b"":
            time.sleep(5)
            dual_sock.reconnect()
            continue

        cmd, data = parse_protocol(pack)
        # ack_sock(received_sock, cmd_type=cmd)
        response, result = process_cmd(cmd=cmd, data=data, connected_sock=dual_sock, detector=detector, settings=settings)

        if result != "":
            database.add_data(result)



if __name__ == '__main__':
    # 2个端口
    # 接受端口21122
    # 发送端口21123
    # 接收到图片 n_rows * n_bands * n_cols, float32
    # 发送图片 n_rows * n_cols, uint8
    main(is_debug=False)
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
