# 7.15
移植了EG_UNet项目中部分代码，新增了UNet的基础模型

# 7.16
新建了一个损失函数，根据NDVI指数来确定更感兴趣的部分。如果NDVI指数在0~0.5之间的话，加大对这些区域的惩罚，使模型更好地分割这些区域
修改了train.py，使其能够保存最后一轮训练的模型
修改了eval.py，让其能够挑选last.pt和best.pt两个权重文件进行训练

# 7.17
新增了一个模型utransform,在UNet的中间卷积层添加了自注意力模块  
修改了train.py,使其能适配utransform,同时改进了保存last_model的逻辑
修改了eval.py,使其能适配utransform

