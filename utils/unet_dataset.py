import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from osgeo import gdal

# 读取txt文件，返回图像路径列表和标签路径列表
def read_txt(path):
    ims, labels = [], []
    # 以只读模式打开txt文件
    with open(path, 'r') as f:
        # 逐行读取文件内容
        for line in f.readlines():
            # 去除每行首尾的空白字符，并按空格分割成图像路径和标签路径
            im, label = line.strip().split()
            ims.append(im)
            labels.append(label)
    return ims, labels

# 读取标签文件（TIFF格式），返回标签数据数组
def read_label(filename):
    # 打开TIFF文件
    dataset = gdal.Open(filename)
    # 获取栅格矩阵的列数
    im_width = dataset.RasterXSize
    # 获取栅格矩阵的行数
    im_height = dataset.RasterYSize
    # im_geotrans = dataset.GetGeoTransform() # 仿射矩阵
    # im_proj = dataset.GetProjection() # 地图投影信息
    # 将数据读取为数组，对应栅格矩阵
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))
    # 释放数据集对象，避免内存泄漏
    del dataset
    return im_data

# 读取TIFF格式的图像文件，根据训练模式进行归一化处理
def read_tiff(filename, train=True):
    # 打开TIFF文件
    dataset = gdal.Open(filename)
    # 获取栅格矩阵的列数
    im_width = dataset.RasterXSize
    # 获取栅格矩阵的行数
    im_height = dataset.RasterYSize
    # im_geotrans = dataset.GetGeoTransform() # 仿射矩阵
    # im_proj = dataset.GetProjection() # 地图投影信息
    # 将数据读取为数组，对应栅格矩阵
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    if train:
        # 对不同波段进行归一化处理
        im_data[1, ...] = im_data[1, ...] * 255 / 1375
        im_data[2, ...] = im_data[2, ...] * 255 / 1583
        im_data[3, ...] = im_data[3, ...] * 255 / 1267
        im_data[4, ...] = im_data[4, ...] * 255 / 2612
        im_data[0, ...] = im_data[0, ...] * 255 / 122
    else:
        # 测试模式下同样进行归一化处理
        im_data[1, ...] = im_data[1, ...] * 255 / 1375
        im_data[2, ...] = im_data[2, ...] * 255 / 1583
        im_data[3, ...] = im_data[3, ...] * 255 / 1267
        im_data[4, ...] = im_data[4, ...] * 255 / 2612
        im_data[0, ...] = im_data[0, ...] * 255 / 122
    # 释放数据集对象，避免内存泄漏
    del dataset
    return im_data

# 将20类标签转换为7类标签
def class_7(filename):
    # 读取TIFF格式的标签文件
    label = np.array(read_tiff(filename))
    label_7 = label
    # 遍历标签数组的每个元素
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] in range(0, 3):
                label_7[i][j] = 0
            elif label[i][j] in range(3, 7):
                label_7[i][j] = 1
            elif label[i][j] in range(7, 11):
                label_7[i][j] = 2
            elif label[i][j] in range(11, 13):
                label_7[i][j] = 3
            elif label[i][j] in range(13, 16):
                label_7[i][j] = 4
            elif label[i][j] in range(16, 19):
                label_7[i][j] = 5
            elif label[i][j] == 19:
                label_7[i][j] = 6
    return label_7

# 自定义数据集类，继承自torch.utils.data.Dataset
class UnetDataset(Dataset):
    def __init__(self, txtpath, transform, train=True):
        super().__init__()
        # 读取txt文件，获取图像路径列表和标签路径列表
        self.ims, self.labels = read_txt(txtpath)
        # 图像变换操作
        self.transform = transform
        # 训练模式标志
        self.train = train

    def __getitem__(self, index):
        root_dir = 'dataset'
        # 拼接图像文件的完整路径
        im_path = os.path.join(root_dir, self.ims[index])
        # 拼接标签文件的完整路径
        label_path = os.path.join(root_dir, self.labels[index])
        if_train = self.train
        # 读取图像文件
        image = read_tiff(im_path, if_train)
        image = np.array(image)
        # 调整图像数组的维度顺序
        image = np.transpose(image, (1, 2, 0))
        # 将图像数组转换为torch.Tensor
        image = transforms.ToTensor()(image)
        # 将图像数据类型转换为float32，并移动到GPU上
        image = image.to(torch.float32).cuda()
        # 对图像进行变换操作，并移动到GPU上
        image = self.transform(image).cuda()
        # 读取标签文件，并将其转换为torch.Tensor，数据类型为int64，并移动到GPU上
        label = torch.from_numpy(np.asarray(read_label(label_path), dtype=np.int32)).long().cuda()
        return image, label, label_path

    def __len__(self):
        # 返回数据集的长度
        return len(self.ims)