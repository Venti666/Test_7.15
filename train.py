import time
import os
import logging
from tqdm import tqdm

from utils import unet_dataset
from models import unetFEGcn,unet,utransform
from metrics import eval_metrics
from utils.losses import CustomNDVILoss  # 导入自定义损失函数
# from predict import predict
# from lr_schedule import step_lr, exp_lr_scheduler

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# 设置可见的 CUDA 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(config):
    # 训练配置
    # 指定使用的设备为 CUDA 设备 0
    device = torch.device('cuda:0')
    # 从配置文件中选择要训练的模型
    selected = config['train_model']['model'][config['train_model']['select']]
    if selected  == 'unetFEGcn':
        # 初始化 unetFEGcn 模型
        model = unetFEGcn.UNet(num_classes=config['num_classes'])
    elif selected == 'unet':
        # 初始化 UNet 模型
        model = unet.UNet(num_classes=config['num_classes'])
    elif selected == 'utransform':
        # 初始化 UMamba 模型
        model = utransform.UTransform(num_classes=config['num_classes'])

    # 将模型移动到指定设备上
    model.to(device)

    # 初始化日志记录器
    logger = initLogger(selected)

    # 定义自定义损失函数
    criterion = CustomNDVILoss(alpha=2.0)  # 可以调整 alpha 值

    # 训练数据处理
    # 定义数据归一化的转换操作
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.209,0.394,0.380,0.344,0.481],std=[0.141,0.027,0.032,0.046,0.069])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]
    )
    # 创建训练数据集
    dst_train = unet_dataset.UnetDataset(config['train_list'], transform=transform,train=True)
    # 创建训练数据加载器
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config['batch_size'])

    # 验证数据处理
    # 定义验证数据的归一化转换操作
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.209,0.394,0.380,0.344,0.481],std=[0.141,0.027,0.032,0.046,0.069])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]
    )
    # 创建验证数据集
    dst_valid = unet_dataset.UnetDataset(config['test_list'], transform=transform,train=False)
    # 创建验证数据加载器
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config['batch_size'])

    cur_acc = []
    # 定义优化器为 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=[config['momentum'], 0.999], weight_decay=config['weight_decay'])
    # 初始化验证集的最大像素准确率和最小损失
    val_max_pixACC = 0.0
    val_min_loss = 100.0

    # 新增：用于存储最后十轮的最佳模型
    best_last_model = None

    for epoch in range(config['num_epoch']):
        # 记录每个 epoch 的开始时间
        epoch_start = time.time()
        # lr

        # 将模型设置为训练模式
        model.train()
        # 初始化损失总和
        loss_sum = 0.0
        # 初始化正确预测的像素总数
        correct_sum = 0.0
        # 初始化标记的像素总数
        labeled_sum = 0.0
        # 初始化交集的像素总数
        inter_sum = 0.0
        # 初始化并集的像素总数
        unoin_sum = 0.0
        # 初始化像素准确率
        pixelAcc = 0.0
        # 初始化交并比
        IoU = 0.0
        # 创建训练进度条
        tbar = tqdm(dataloader_train, ncols=120)

        # 初始化训练集的混淆矩阵
        conf_matrix_train = np.zeros((config['num_classes'],config['num_classes']))

        for batch_idx, (data, target, path) in enumerate(tbar):
            # 记录每个批次的开始时间
            tic = time.time()

            data, target = data.to(device), target.to(device)
            # 清空优化器的梯度
            optimizer.zero_grad()
            # 前向传播，得到模型的输出
            output = model(data)
            # 计算损失
            loss = criterion(output, target, data)  # 使用自定义损失函数
            # 累加损失
            loss_sum += loss.item()
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

            # 计算准确率、标记像素数、交集、并集和混淆矩阵
            correct, labeled, inter, unoin, conf_matrix_train = eval_metrics(output, target, config['num_classes'],conf_matrix_train)
            # 累加正确预测的像素数
            correct_sum += correct
            # 累加标记的像素数
            labeled_sum += labeled
            # 累加交集的像素数
            inter_sum += inter
            # 累加并集的像素数
            unoin_sum += unoin
            # 计算像素准确率
            pixelAcc = 1.0 * correct_sum / (np.spacing(1)+labeled_sum)
            # 计算交并比
            IoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
            # 更新进度条的描述信息
            tbar.set_description('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum/((batch_idx+1)*config['batch_size']),
                pixelAcc, IoU.mean(),
                time.time()-tic, time.time()-epoch_start))
            # 记录当前的像素准确率
            cur_acc.append(pixelAcc)

        # 记录训练集的日志信息
        logger.info('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} IOU {}  mIoU {:.5f} '.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(IoU), IoU.mean()))

        # 验证阶段
        # 记录验证阶段的开始时间
        test_start = time.time()

        # 将模型设置为评估模式
        model.eval()
        # 初始化验证集的损失总和
        loss_sum = 0.0
        # 初始化验证集的正确预测像素总数
        correct_sum = 0.0
        # 初始化验证集的标记像素总数
        labeled_sum = 0.0
        # 初始化验证集的交集像素总数
        inter_sum = 0.0
        # 初始化验证集的并集像素总数
        unoin_sum = 0.0
        # 初始化验证集的像素准确率
        pixelAcc = 0.0
        # 初始化验证集的平均交并比
        mIoU = 0.0
        # 创建验证进度条
        tbar = tqdm(dataloader_valid, ncols=120)
        # 初始化每一类的精确率
        class_precision=np.zeros(config['num_classes'])
        # 初始化每一类的召回率
        class_recall=np.zeros(config['num_classes'])
        # 初始化每一类的 F1 分数
        class_f1=np.zeros(config['num_classes'])
        # val_list=[]

        with torch.no_grad():
            # 初始化验证集的混淆矩阵
            conf_matrix_val = np.zeros((config['num_classes'],config['num_classes']))
            for batch_idx, (data, target, path) in enumerate(tbar):
                # 记录每个验证批次的开始时间
                tic = time.time()

                data, target = data.to(device), target.to(device)
                # 前向传播，得到模型的输出
                output = model(data)
                # 计算损失
                loss = criterion(output, target, data)  # 使用自定义损失函数
                # 累加验证集的损失
                loss_sum += loss.item()

                # 计算准确率、标记像素数、交集、并集和混淆矩阵
                correct, labeled, inter, unoin, conf_matrix_val = eval_metrics(output, target, config['num_classes'], conf_matrix_val)
                # 累加验证集的正确预测像素数
                correct_sum += correct
                # 累加验证集的标记像素数
                labeled_sum += labeled
                # 累加验证集的交集像素数
                inter_sum += inter
                # 累加验证集的并集像素数
                unoin_sum += unoin
                # 计算验证集的像素准确率
                pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                # 计算验证集的平均交并比
                mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

                for i in range(config['num_classes']):
                    # 计算每一类的精确率
                    class_precision[i]=1.0*conf_matrix_val[i,i]/conf_matrix_val[:,i].sum()
                    # 计算每一类的召回率
                    class_recall[i]=1.0*conf_matrix_val[i,i]/conf_matrix_val[i].sum()
                    # 计算每一类的 F1 分数
                    class_f1[i]=(2.0*class_precision[i]*class_recall[i])/(class_precision[i]+class_recall[i])

                # 更新验证进度条的描述信息
                tbar.set_description('VAL ({}) | Loss: {:.5f} | Acc {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                    epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                    pixelAcc, mIoU.mean(),
                    time.time() - tic, time.time() - test_start))
            if loss_sum < val_min_loss:
                # 更新最小验证损失
                val_min_loss = loss_sum
                # 记录最佳 epoch 和混淆矩阵的总和
                best_epoch =np.zeros(2)
                best_epoch[0]=epoch
                best_epoch[1]=conf_matrix_val.sum()
                # 如果保存模型的路径不存在，则创建该路径
                if os.path.exists(config['save_model']['save_path']) is False:
                    os.mkdir(config['save_model']['save_path'])
                # 保存模型的状态字典
                torch.save(model.state_dict(), os.path.join(config['save_model']['save_path'], selected+'_jx_best.pth'))
                # 保存验证集的混淆矩阵
                np.savetxt(os.path.join(config['save_model']['save_path'],  selected+'_conf_matrix_val.txt'),conf_matrix_val,fmt="%d")
                # 保存最佳 epoch 的信息
                np.savetxt(os.path.join(config['save_model']['save_path'], selected+'_best_epoch.txt'),best_epoch)
            
            # 新增：在最后十轮中，保存 OA 和 mIoU 表现最好的模型
            if epoch >= config['num_epoch'] - 10:
                current_model_info = {
                    'epoch': epoch,
                    'val_pixelAcc': pixelAcc,
                    'val_mIoU': mIoU.mean(),
                    'state_dict': model.state_dict().copy(),
                    'conf_matrix_val': conf_matrix_val.copy()
                }

                if best_last_model is None:
                    best_last_model = current_model_info
                else:
                    # 比较当前模型和之前的最佳模型，选择 OA + mIoU 更高的
                    current_score = 0.5*current_model_info['val_pixelAcc'] + current_model_info['val_mIoU']
                    best_score = 0.5*best_last_model['val_pixelAcc'] + best_last_model['val_mIoU']
                    if current_score > best_score:
                        best_last_model = current_model_info
        # 记录验证集的日志信息
        logger.info('VAL ({}) | Loss: {:.5f} | OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(mIoU), mIoU.mean(),toString(class_precision),toString(class_recall),toString(class_f1)))
        
        # 训练结束后，保存最后十轮中表现最好的模型
        if best_last_model is not None:
            torch.save(best_last_model['state_dict'], 
                    os.path.join(config['save_model']['save_path'], 
                    f"{selected}_jx_last.pth"))
            np.savetxt(os.path.join(config['save_model']['save_path'], 
                    f"{selected}_conf_matrix_val_last.txt"), 
                    best_last_model['conf_matrix_val'], fmt="%d")

def toString(IOU):
    # 将 IOU 数组转换为字符串格式
    result = '{'
    for i, num in enumerate(IOU):
        result += str(i) + ': ' + '{:.4f}, '.format(num)

    result += '}'
    return result

def initLogger(model_name):
    # 初始化日志记录器
    logger = logging.getLogger()
    # 设置日志级别为 INFO
    logger.setLevel(logging.INFO)

    # 获取当前时间并格式化为字符串
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    # 定义日志保存的路径
    log_path = r'logs'
    # 定义日志文件的名称
    log_name = os.path.join(log_path, "new"+model_name + '_jx_new_metrics' + rq + '.log')
    # 定义日志文件的路径
    logfile = log_name
    # 创建文件处理器
    fh = logging.FileHandler(logfile, mode='w')
    # 设置文件处理器的日志级别为 INFO
    fh.setLevel(logging.INFO)

    # 定义日志的格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # 设置文件处理器的日志格式
    fh.setFormatter(formatter)
    # 将文件处理器添加到日志记录器中
    logger.addHandler(fh)

    return logger

if __name__ == '__main__':
    # train()
    while True:
        print(1)