# -*- coding: utf-8 -*-
"""
Created on Nov 3 21:18:26 2020

@author: l.z.y
@e-mail: li.zhenye@qq.com
"""
import logging
import sys
from typing import Optional

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import binom
import matplotlib.pyplot as plt
import time
import pickle
import os

import config

sys.path.append(os.getcwd())
from root_dir import ROOT_DIR
import utils

FEATURE_INDEX = [0, 1, 2]
delete_columns = 10        # 已弃用

class WoodClass(object):
    def __init__(self, load_from=None, w=2048, h=12450, n=5000, p1=0.3, pur=0.99999, left_correct=False, single_pick_mode=False,
                 debug_mode=False):
        """
        初始化.

        :param w: 图像的尺寸w
        :param h: 图像的尺寸h
        :param p1: 木板色彩在图像中的比例p1
        :param n: 采集用于识别的样本点的个数
        :param single_pick_mode: 是否使用单点提取方案
        """
        if load_from is None:
            if w is None or h is None:
                print("It will damage your performance if you don't set w and h and use single_pick_mode!")
                raise ValueError("w or h is None")
            self.pur, self.p1, self.k = pur, p1, 1
            self.w, self.h, self.n = w, h, n
            self.ww, self.hh = None, None
            self.width = None
            self._single_pick = single_pick_mode
            self.set_purity(self.pur)
            self.change_pick_mode(single_pick_mode)
            self.model = LogisticRegression(C=1e5)
            self.left_correct = left_correct
            # self.model = KNeighborsClassifier()
        else:
            self.load(load_from)
        self.isCorrect = False
        self.correct_color = None
        self.log = utils.Logger(is_to_file=debug_mode)
        self.debug_mode = debug_mode
        self.image_num = 0

    def change_pick_mode(self, single_pick_mode):
        """
        更改图像提取方法，

        :param single_pick_mode:若True， 则为单点随机抽取模式
        :return: None
        """
        w, h, n = self.w, self.h, self.n

        if single_pick_mode:
            self._single_pick = True
            width = int(np.floor(np.sqrt(w * h / n)))
            w0, h0 = np.arange(0, w-width, width), np.arange(0, h-width, width)
            self.ww, self.hh = np.meshgrid(w0, h0)
            self.width = width
        else:
            self._single_pick = False
            ratio = np.sqrt(n / (w * h))
            self.ww, self.hh = int(ratio * w), int(ratio * h)

    def get_rand_sample(self, x):
        """
        在图像中进行随机抽取，如果single_pick_mode为True，则为真随机抽取，反之为假的抽取.

        :param x:
        :return:
        """
        if self._single_pick:
            offset_w, offset_h = np.random.randint(0, self.width), np.random.randint(0, self.width)
            sample = x[self.hh+offset_h, self.ww+offset_w,  ...]
        else:
            sample = cv2.resize(x, (self.ww, self.hh))
        return sample

    def fit_pictures(self, data_path=ROOT_DIR, file_name=None):
        """
        根据给出的data_path 进行 fit.如果没有给出data目录，那么将会使用当前文件夹
        :param data_path:
        :return:
        """
        # 训练数据文件位置
        result = self.get_train_data(data_path, plot_2d=True)
        if result is False:
            return 0
        x, y = result
        score = self.fit(x, y)
        print('model score', score)
        model_name = self.save(file_name)
        return model_name

    def fit(self, x,  y, test_size=0.1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        pre_score = accuracy_score(y_test, y_pred)
        self.log.log("Test accuracy is:"+str(pre_score * 100) + "%.")
        y_pred = self.model.predict(x_train)
        pre_score = accuracy_score(y_train, y_pred)
        self.log.log("Train accuracy is:"+str(pre_score * 100) + "%.")
        y_pred = self.model.predict(x)
        pre_score = accuracy_score(y, y_pred)
        self.log.log("Total accuracy is:"+str(pre_score * 100) + "%.")
        return int(pre_score*100)

    def calculate_p1(self, x, remove_background=False):
        """

        :param x:
        :param remove_background:
        :return:
        """
        if remove_background:
            x = self.remove_background(x)
        kmeans = KMeans(n_clusters=2, init='k-means++')
        kmeans.fit(x)
        result = kmeans.predict(x)  # 聚类结果

    def predict(self, img):
        """
        :param img: 输入图像
        :return: 分类值
        """
        img = self.realtime_correct(img, 10, 20)
        if self.debug_mode:
            cv2.imwrite(str(self.image_num) + ".bmp", img)
            self.image_num += 1
        feature = self.extract_feature(img, remove_background=False, debug_mode=False)
        feature = feature.reshape(1, -1)[:, FEATURE_INDEX]
        if self.isCorrect:
            feature = feature / (self.correct_color+1e-4)
        plt.figure()
        plt.scatter(feature[:, 0], feature[:, 1])
        plt.show()
        pred_color = self.model.predict(feature)
        if self.debug_mode:
            self.log.log(feature)
        return int(pred_color[0])

    def correct(self, img=None, img_path=None):
        """
        记录校准值，将校准值记录到类别内
        :param img: 校准板图片
        :param img_path: 用于校准的图片路径
        :return: 0 if correct success, 1 if failed
        """
        if img is None:
            path = os.path.join(ROOT_DIR, "data", "correct")
            utils.mkdir_if_not_exist(path)
            file_list = os.listdir(path)
            if len(file_list) == 0:
                return 1
            if img_path is None:
                file_name = os.path.join(ROOT_DIR, "data", "correct", file_list[-1])
            else:
                file_name = img_path
            img = cv2.imread(file_name)

        feature = self.extract_feature(img)[FEATURE_INDEX]
        self.correct_color = feature
        self.isCorrect = True
        self.log.log("Correct Successfully!")
        return 0

    def set_purity(self, purity):
        self.pur = purity
        vs_pur = 1 - self.pur
        for i in range(self.n):
            vs_pur_i = binom.cdf(k=i, p=self.p1, n=self.n)
            if vs_pur_i > vs_pur:
                self.k = i
                return i

    def remove_background(self, x):
        # TODO: 利用色度？饱和度或者明度？亮度？去除背景
        # 去背景的方法效果不好，太慢了，所以没弄了。。。
        # x = x[2000:16000, 300:1600, :]
        x = x[2000:10000, 300:1600, :]
        return x

    def save(self, file_name=None):
        """
        保存当前文件下的classify.model文件模型
        save_parameters 为要保存的参数
        :return: None
        """
        if file_name is None:
            file_name = "model_" + time.strftime("%Y-%m-%d_%H-%M") + ".p"
            file_name = os.path.join(ROOT_DIR, "models", file_name)
        model_dic = {"n": self.n, "k": self.k, "p1": self.p1, "pur": self.pur, "model": self.model,
                     "ww": self.ww, "hh": self.hh, "width": self.width, "w": self.w, "h": self.h,
                     "mode": self._single_pick, "isCorrect": self.isCorrect, "left_correct": self.left_correct}
        with open(file_name, "wb") as f:
            pickle.dump(model_dic, f)
        self.log.log("Save file to '" + str(file_name) + "'")
        return file_name

    def load(self, path=None):
        if path is None:
            path = os.path.join(ROOT_DIR, "models")
            utils.mkdir_if_not_exist(path)
            model_files = os.listdir(path)
            if len(model_files) == 0:
                self.log.log("No model found!")
                return 1
            self.log.log("./ Models Found:")
            _ = [self.log.log("├--"+str(model_file)) for model_file in model_files]
            file_times = [model_file[6:-2] for model_file in model_files]
            latest_model = model_files[int(np.argmax(file_times))]
            self.log.log("└--Using the latest model: "+str(latest_model))
            path = os.path.join(ROOT_DIR, "models", str(latest_model))
        if not os.path.isabs(path):
            logging.warning('给的是相对路径')
            return -1
        if not os.path.exists(path):
            logging.warning('文件不存在')
            return -1
        with open(path, "rb") as f:
            model_dic = pickle.load(f)
        self.n, self.k, self.p1, self.pur = model_dic["n"], model_dic["k"], model_dic["p1"], model_dic["pur"]
        self.ww, self.hh, self.width = model_dic["ww"], model_dic["hh"], model_dic["width"]
        self.w, self.h, self.model = model_dic["w"], model_dic["h"], model_dic["model"]
        self.isCorrect = model_dic["isCorrect"]
        self._single_pick = model_dic["mode"]
        self.set_purity(self.pur)
        self.change_pick_mode(self._single_pick)
        self.left_correct = model_dic["left_correct"]
        return 0

    def extract_feature(self, x, correct_color=False, remove_background=False, debug_mode=False):
        """
        获取图片的特征,色彩值的mean和var【l, a, b, s_l, s_a, s_b】.
        :param x: 图片
        :param correct_color: 是否进行颜色校准
        :param remove_background:是否需要移除背景
        :param debug_mode: 是否使用debug模式
        :return:
        """
        if remove_background:
            x = self.remove_background(x)
        x = self.get_rand_sample(x)
        if correct_color is True:
            x = x / self.correct_color
        if debug_mode:
            plt.figure()
            plt.subplot(211)
            plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        x_hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        x = np.concatenate((x, x_hsv), axis=2)
        x = np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2]))
        hist, bins = np.histogram(x[:, 0], bins=10)
        # x = x[np.argsort(x[:, 0])]
        # x = x[-self.k:, :]
        hist_number = np.argmax(hist)
        x = x[(x[:, 0] > bins[hist_number]) & (x[:, 0] < bins[hist_number+1]), :]
        if debug_mode:
            # self.log.log(x)
            self.log.log(x.shape)
            # self.log.log(self.k)
            # self.log.log(x)
            self.log.log(x.shape)
        mean_value = np.mean(x, axis=0)
        if debug_mode:
            self.log.log("mean color:"+str(mean_value))
            plt.subplot(212)
            color_img = np.asarray(np.ones((100, 100, 3), dtype=np.uint8) * mean_value[:3], dtype=np.uint8)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_LAB2RGB)
            plt.imshow(color_img)
            plt.show()
        var_value = np.var(x, axis=0)
        feature = np.hstack((mean_value, var_value))
        if debug_mode:
            self.log.log("var: "+str(var_value))
        return feature

    def get_image_data(self, img_dir="./data/dark"):
        """
        :param img_dir: 图像文件的路径
        :return: 图像数据
        """
        img_data = []
        utils.mkdir_if_not_exist(img_dir)
        files = os.listdir(img_dir)
        if len(files) == 0:
            return False
        for file in files:
            path = os.path.join(img_dir, file)
            if self.debug_mode:
                self.log.log(path)
            train_img = cv2.imread(path)
            train_img = self.realtime_correct(train_img, 10, 20)
            data = self.extract_feature(train_img)
            img_data.append(data)
        img_data = np.array(img_data)
        return img_data

    def get_train_data(self, data_dir=None, plot_2d=False, plot_data_3d=False, save_data=False):
        """
        获取图像数据
        :return: x_data, y_data
        """
        data_dir = os.path.join(ROOT_DIR, "data", "data20220919") if data_dir is None else data_dir
        dark_data = self.get_image_data(img_dir=os.path.join(data_dir, "dark"))
        middle_data = self.get_image_data(img_dir=os.path.join(data_dir, "middle"))
        light_data = self.get_image_data(img_dir=os.path.join(data_dir, "light"))
        if (dark_data is False) or (middle_data is False) or (light_data is False):
            return False
        x_data = np.vstack((dark_data, middle_data, light_data))
        dark_label = np.zeros(len(dark_data)).T
        middle_label = np.ones(len(middle_data)).T
        light_label = 2 * np.ones(len(light_data)).T
        y_data = np.hstack((dark_label, middle_label, light_label))
        x_data = x_data[:, FEATURE_INDEX]
        # 进行色彩数据校正
        if self.isCorrect:
            x_data = x_data / (self.correct_color+1e-4)
        if plot_data_3d:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.scatter(x_data[:, 1], x_data[:, 2], x_data[:, 0], c=y_data, edgecolors="k")
            ax.set_xlabel("a*")
            ax.set_ylabel("b*")
            ax.set_zlabel("l")
            plt.show()
        if plot_2d:
            plt.figure()
            plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
            plt.show()
            # 尝试最合适的特征组合，保存提取出的特征的方法
            # 0: l, 1: a, 2: b, 3: var(l), 4: var(a), 5: var(s), 6: h,  7: s, 8: v, 9: var(h) 10: var(s): 11: var(v)
            # 全部：0.941
            # [:, [0, 1, 2, 3, 4, 5, 6, 7]] : 0.88
            # [:, [0, 1, 2]] : 0.911
            # [:, [0, 1, 2, 6, 7, 8]] : 0.941
            # [:, [1, 2, 6, 7, 8]] : 0.9705
            # [:, [1, 2, 6, 7]] : 0.9705
            # [:, [1, 2, 4, 5, 6, 7]] : 0.941
            # [:, [0, 1, 2, 6, 7]] : 0.8529
        if save_data:
            with open(os.path.join("data", "data.p"), "rb") as f:
                pass
        return x_data, y_data


    def realtime_correction(self, img):
        """
        实时校正
        :param img:
        :return:
        """
        # 按照第一列的标准进行矫正并去除前若干列
        standard_img = np.ones_like(img) * 255
        img_like = np.empty_like(img)
        img_like[:, :, :] = img[:, 0:1, :]
        img = (img / img_like * standard_img)[:, delete_columns:, :]
        return img

    def realtime_correct(self, img:np.ndarray, correct_col_num: int, cut_col_num: Optional[int] = None,
                         standard_color: Optional[tuple] = (255, 255, 255)) -> np.ndarray:
        """
        实时利用左侧边的correct_col_num列进行色彩校正

        :param img: 待校正的图片 shape = (n_rows, n_cols, n_channels)
        :param correct_col_num: 用于矫正的列数
        :param cut_col_num: 最终要去除的列数, 默认为None, 表示去除correct_col_num列
        :param standard_color: 标准色彩, 默认为白色
        :return: 校正后的图片 shape = (n_rows, n_cols - cut_col_num, n_channels)
        """
        if self.left_correct:
            # 按照correct_col_num列数量取出最左侧校正板区域成像结果
            correct_img = img[:, :correct_col_num, :]
            # 校正区域进行均值化
            correct_scaler = np.mean(correct_img, axis=1) / np.array(standard_color)
            # 校正区域的均值除以原图的均值
            if cut_col_num is None:
                cut_col_num = correct_col_num
            img = img[:, cut_col_num:, :] / correct_scaler[:, np.newaxis, :]
            img = np.clip(img, 0, 255).astype(dtype=np.uint8)
        return img


if __name__ == '__main__':
    from config import Config
    settings = Config()
    # 初始化wood
    wood = WoodClass(w=4096, h=1200, n=12500, p1=0.4, debug_mode=False)
    print("色彩纯度控制量{}/{}".format(wood.k, wood.n))
    data_path = settings.data_path
    # wood.correct()
    # wood.load()
    # fit 相应的文件夹
    settings.model_path = str(ROOT_DIR / 'models' / wood.fit_pictures(data_path=data_path))

    # 测试单张图片的预测，predict_mode=True表示导入本地的model, False为现场训练的
    pic = cv2.imread(r"data/duizhao/rgb7.png")
    start_time = time.time()
    # for i in range(100):
    wood_color = wood.predict(pic)
    end_time = time.time()
    print("time consume:"+str((end_time - start_time)/100))
    print("wood_color:"+str(wood_color))

