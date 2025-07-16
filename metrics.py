import torch
import numpy as np

# 该函数用于评估模型的预测结果，计算预测的准确性、交并比（IoU）等指标，并更新混淆矩阵
def eval_metrics(output, target, num_classes, conf_matrix):
    # 从模型输出中获取预测类别，output.max(1) 返回每行的最大值及其索引，这里取索引作为预测类别
    _, predict = output.max(1)

    # 将预测结果和真实标签从GPU移动到CPU，并展平成一维数组，方便后续处理
    c_predict = predict.cpu().numpy().flatten()
    c_target = target.cpu().numpy().flatten()

    # 遍历预测结果和真实标签，更新混淆矩阵
    # 混淆矩阵的行表示真实类别，列表示预测类别，矩阵元素表示预测为该列类别且真实为该行类别的样本数量
    for i in range(len(c_predict)):
        conf_matrix[c_target[i], c_predict[i]] += 1

    # 为了后续计算方便，将预测结果和真实标签加1，避免出现类别为0的情况
    predict = predict.long() + 1
    target = target.long() + 1

    # 计算有标签的像素数量，即真实标签大于0的像素数量
    pixel_labeled = (target > 0).sum()
    # 计算预测正确的像素数量，即预测结果等于真实标签且真实标签大于0的像素数量
    pixel_correct = ((predict == target) * (target > 0)).sum()

    # 只考虑有标签的像素，将无标签的像素对应的预测结果置为0
    predict = predict * (target > 0).long()
    # 计算预测结果和真实标签的交集，即预测正确的有标签像素
    intersection = predict * (predict == target).long()

    # 计算每个类别的交集面积，使用torch.histc函数统计每个类别的像素数量
    area_inter = torch.histc(intersection.float(), num_classes, 1, num_classes)
    # 计算每个类别的预测面积，即预测为该类别的像素数量
    area_pred = torch.histc(predict.float(), num_classes, 1, num_classes)
    # 计算每个类别的真实面积，即真实为该类别的像素数量
    area_lab = torch.histc(target.float(), num_classes, 1, num_classes)
    # 计算每个类别的并集面积，根据公式：并集面积 = 预测面积 + 真实面积 - 交集面积
    area_union = area_pred + area_lab - area_inter

    # 将计算结果从GPU移动到CPU，并四舍五入保留5位小数
    correct = np.round(pixel_correct.cpu().numpy(), 5)
    labeld = np.round(pixel_labeled.cpu().numpy(), 5)
    inter = np.round(area_inter.cpu().numpy(), 5)
    union = np.round(area_union.cpu().numpy(), 5)

    # 这里注释掉的代码是计算像素准确率（pixacc）和平均交并比（mIoU）的公式
    # pixacc = 1.0 * correct / (np.spacing(1) + labeld)
    # mIoU = (1.0 * inter / (np.spacing(1) + union)).mean()

    # 返回计算得到的正确像素数量、有标签像素数量、每个类别的交集面积、每个类别的并集面积以及更新后的混淆矩阵
    return correct, labeld, inter, union, conf_matrix