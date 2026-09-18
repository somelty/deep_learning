from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch

class iris_dataloader(Dataset):
    '''
    Load iris dataset,一个数据集加载的类，在自定义加载类时，也就是继承 Dataset 时要求重写三个方法：init，len，getitem
    '''
    def __init__(self, data_path):
        '''
        主要设置数据集路径和加载数据集
        :param data_path:
        '''
        self.data_path = data_path

        assert os.path.exists(self.data_path), "data path does not exist"

        # 读取数据，并添加列名
        df = pd.read_csv(self.data_path, names=[0, 1, 2, 3, 4])

        # 神经网络不会直接输出像 setosa，需要进行映射
        d = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        df[4] = df[4].map(d) # 第五列花朵名字按照 d 字典规则进行映射

        # 取出数据
        data = df.iloc[:, :4]
        label = df.iloc[:, 4:]

        data = (data - data.mean()) / data.std() # z值化，稳定输入利于性能
        # 刚才用pd加载文件，默认数据类型df，pytorch只接收tensor张量类型，要进行转换
        self.data = torch.from_numpy(np.array(data, dtype=np.float32))
        # 这里之前写的 int8报错（RuntimeError: expected target dtype to be Long or Byte, but got Char），CrossEntropy 要 long (int64)
        # 数组索引，在 PyTorch 底层要求 64 位整数 (long)，32 位 int、8 位 int 都不能当做索引
        self.label = torch.from_numpy(np.array(label, dtype=np.int64))

        self.data_num = len(label)
        print("data size: ", self.data_num)


    # 将来进行训练可能会对数据集划分生成不同批量数据，pytorch 需要知道数据集大小才可以进行划分
    # 告诉 DataLoader 该数据集有多大，以便计算批次数量
    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        '''
        函数目的是通过索引来获取数据
        按索引读取单个样本，它返回的是一个样本（如 (data, label) 元组）
        '''
        # 张量本身支持下标索引，不需要每次取值都把整个数据集转成 list
        return self.data[index], self.label[index]




