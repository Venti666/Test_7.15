import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import unetFEGcn,unet
from utils.unet_dataset import read_tiff
from osgeo import gdal
from metrics import eval_metrics
from train import toString
import os
from metrics import eval_metrics
import numpy as np
import torch
from torchvision import transforms

# 读取标签文件的函数
def read_label(filename):
    # 打开文件
    dataset = gdal.Open(filename)
    # 获取栅格矩阵的列数
    im_width = dataset.RasterXSize
    # 获取栅格矩阵的行数
    im_height = dataset.RasterYSize

    # 读取数据并将其转换为数组
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # 删除数据集对象以释放内存
    del dataset
    return im_data

# 评估模型的函数
def eval(config):
    # 指定使用的设备为GPU 0
    device = torch.device('cuda:0')
    # 设置环境变量，指定可见的GPU设备为0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 从配置文件中选择要使用的模型
    selected = config['train_model']['model'][config['train_model']['select']]
    # 如果选择的模型是unetFEGcn
    if selected == 'unetFEGcn':
        # 初始化unetFEGcn模型，并指定类别数量
        model = unetFEGcn.UNet(num_classes=config['num_classes'])
    elif selected == 'unet':
        # 初始化UNet模型，并指定类别数量
        model = unet.UNet(num_classes=config['num_classes'])

    # 加载模型的检查点文件
    check_point = os.path.join(config['save_model']['save_path'], selected + '_jx_last.pth')
    check_point = os.path.join(config['save_model']['save_path'], selected + '_jx_best.pth')
    # 定义数据归一化的转换操作
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.209, 0.394, 0.380, 0.344, 0.481], std=[0.141, 0.027, 0.032, 0.046, 0.069])
        ]
    )
    # 加载模型的状态字典
    model.load_state_dict(torch.load(check_point), False)
    # 将模型移动到GPU上
    model.cuda()
    # 将模型设置为评估模式
    model.eval()
    # 初始化测试集的混淆矩阵
    conf_matrix_test = np.zeros((config['num_classes'], config['num_classes']))

    # 初始化正确预测的像素总数
    correct_sum = 0.0
    # 初始化标注的像素总数
    labeled_sum = 0.0
    # 初始化交集的像素总数
    inter_sum = 0.0
    # 初始化并集的像素总数
    unoin_sum = 0.0
    # 初始化像素准确率
    pixelAcc = 0.0
    # 初始化平均交并比
    mIoU = 0.0

    # 初始化每一类的精确率数组
    class_precision = np.zeros(config['num_classes'])
    # 初始化每一类的召回率数组
    class_recall = np.zeros(config['num_classes'])
    # 初始化每一类的F1分数数组
    class_f1 = np.zeros(config['num_classes'])

    # 打开包含测试数据文件名的文本文件
    with open(config['img_txt'], 'r', encoding='utf-8') as f:
        # 逐行读取文件
        for line in f.readlines():
            # 分割每行的图像文件名和标签文件名
            image_name, label_name = line.strip().split()
            # 根目录，这里为空
            root_dir = 'dataset'
            # 拼接图像文件的完整路径
            image_name = os.path.join(root_dir, image_name)
            # 拼接标签文件的完整路径
            label_name = os.path.join(root_dir, label_name)
            # 读取标签文件并将其转换为torch张量，然后移动到GPU上
            label = torch.from_numpy(np.asarray(read_label(label_name), dtype=np.int32)).long().cuda()

            # 读取图像文件
            image = read_tiff(image_name, train=True)
            # 将图像转换为numpy数组
            image = np.array(image)
            # 调整图像数组的维度顺序
            image = np.transpose(image, (1, 2, 0))
            # 将图像转换为torch张量
            image = transforms.ToTensor()(image)
            # 将图像张量转换为浮点型并移动到GPU上
            image = image.to(torch.float32).cuda()
            # 对图像进行归一化处理
            image = transform(image).cuda()
            # 为图像添加一个维度，模拟批量大小为1
            image = image.unsqueeze(0)

            # 使用模型对图像进行预测
            output = model(image)

            # 调用评估指标函数，计算正确预测的像素数、标注的像素数、交集、并集和混淆矩阵
            correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(output, label, config['num_classes'],
                                                                           conf_matrix_test)
            # 累加正确预测的像素数
            correct_sum += correct
            # 累加标注的像素数
            labeled_sum += labeled
            # 累加交集的像素数
            inter_sum += inter
            # 累加并集的像素数
            unoin_sum += unoin
            # 计算像素准确率
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            # 计算平均交并比
            mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

            # 计算每一类的精确率、召回率和F1分数
            for i in range(config['num_classes']):
                # 每一类的精确率
                class_precision[i] = 1.0 * conf_matrix_test[i, i] / conf_matrix_test[:, i].sum()
                # 每一类的召回率
                class_recall[i] = 1.0 * conf_matrix_test[i, i] / conf_matrix_test[i].sum()
                # 每一类的F1分数
                class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])

    # 打印评估指标，包括总体准确率、交并比、平均交并比、每一类的精确率、召回率和F1分数
    print('OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
        pixelAcc, toString(mIoU), mIoU.mean(), toString(class_precision), toString(class_recall), toString(class_f1)))
    # 创建混淆矩阵文件夹（如果不存在）
    if not os.path.exists("confuse_matrix"):
        os.makedirs("confuse_matrix")
    # 将测试集的混淆矩阵保存为文本文件
    np.savetxt(os.path.join("confuse_matrix", selected + '_jx_matrix_test.txt'), conf_matrix_test, fmt="%d")

if __name__ == "__main__":
    # 打开评估配置文件
    with open(r'eval_config.json', encoding='utf-8') as f:
        # 加载配置文件内容
        config = json.load(f)
    # 调用评估函数进行模型评估
    eval(config)