# -*- codeing = utf-8 -*-
# Time : 2022/10/17 11:07
# @Auther : zhouchao
# @File: config.py
# @Software:PyCharm
import json
import os
from pathlib import WindowsPath

from root_dir import ROOT_DIR


class Config(object):
    model_path = ROOT_DIR / 'config.json'

    def __init__(self):
        self._param_dict = {}
        if os.path.exists(Config.model_path):
            self._read()
        else:
            self.model_path = str(ROOT_DIR / 'models/model_2022-10-17_11-10.p')
            self.data_path = str(ROOT_DIR / 'data/data20220919')
            self.database_addr = str("mysql+pymysql://root:@localhost:3306/orm_test")  # 测试用数据库地址
            self._param_dict['model_path'] = self.model_path
            self._param_dict['data_path'] = self.data_path
            self._param_dict['database_addr'] = self.database_addr

    def __setitem__(self, key, value):
        if key in self._param_dict:
            self._param_dict[key] = value
            self._write()

    def __getitem__(self, item):
        if item in self._param_dict:
            return self._param_dict[item]

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if '_param_dict' in self.__dict__ and key != '_param_dict':
            if isinstance(value, WindowsPath):
                value = str(value)
            self.__dict__['_param_dict'][key] = value
            self._write()

    def _read(self):
        with open(Config.model_path, 'r') as f:
            self._param_dict = json.load(f)
            self.data_path = self._param_dict['data_path']
            self.model_path = self._param_dict['model_path']
            self.database_addr = self._param_dict['database_addr']

    def _write(self):
        with open(Config.model_path, 'w') as f:
            json.dump(self._param_dict, f)


if __name__ == '__main__':
    config = Config()
    print(config.model_path)
    print(config.data_path)
