import socket
import numpy as np
import cv2
from classifer import WoodClass
import time


def main():
    model_path = "models/model_2022-09-06_13-08.p"
    socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_receive.connect(('127.0.0.1', 21122))
    socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_send.connect(('127.0.0.1', 21123))
    # _ = socket_receive.recv(4096*1000*10)
    detector = WoodClass(w=4096, h=1200, n=3000, debug_mode=False)
    detector.load(path=model_path)
    while True:
        # receive data
        t1 = time.time()
        size_buff = socket_receive.recv(5)
        if (size_buff[4] == 0):
            n_rows, n_cols = size_buff[0] << 8 | size_buff[1], size_buff[2] << 8 | size_buff[3]
            data_size = n_rows * n_cols * 3
            print(data_size)
            recv_size = data_size
            buff_all, size = [], 0
            while True:
                data_buff = socket_receive.recv(recv_size)
                recv_size -= len(data_buff)
                buff_all += data_buff
                if recv_size == 0:
                    break
            print(len(buff_all))
            raw_data = np.frombuffer(bytes(buff_all), dtype=np.uint8).reshape(int(n_rows), int(n_cols), -1)
            print(raw_data.shape)
            wood_color = detector.predict(raw_data)
            # cv2.imshow("img", raw_data)
            # cv2.waitKey(30)
            # print('Class is ', wood_color)
            if wood_color == 0:
                socket_send.send(b'S')
                print('S send success')
            elif wood_color == 1:
                socket_send.send(b'Z')
                print('Z send success')
            elif wood_color == 2:
                socket_send.send(b'Q')
                print('Q send success')
            print((time.time()-t1))
        elif (size_buff[4] == 1):
            detector = WoodClass(w=4096, h=1200, n=3000, debug_mode=False)
            detector.correct()
            detector.fit_pictures(data_path=r"C:\Users\FEIJINTI\PycharmProjects\wood_color")
            socket_send.send(b'G')
        elif (size_buff[4] == 2):
            pass





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
