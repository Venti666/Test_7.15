import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from osgeo import gdal

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class NDVIAnalyzer:
    def __init__(self, image_dir, label_dir, class_names=None, image_ext='.tif', label_ext='.tif'):
        """初始化NDVI分析器，基于LCMA数据集特性配置"""
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.ndvi_by_class = {}
        
        # 论文中定义的各通道归一化参数（TABLE III）
        self.normalization = {
            "max": [122, 1375, 1583, 1267, 2612],  # 各通道最大值
            "mean": [0.209, 0.394, 0.380, 0.344, 0.398],  # 归一化后均值
            "std": [0.141, 0.027, 0.032, 0.046, 0.108]    # 标准差
        }

    def read_image(self, image_path):
        """读取多光谱影像并按论文标准进行归一化处理"""
        dataset = gdal.Open(image_path)
        if dataset is None:
            print(f"无法打开影像: {image_path}")
            return None, None
        
        # 检查波段数量是否符合LCMA数据集定义（5个通道）
        if dataset.RasterCount != 5:
            print(f"影像{image_path}波段数量不符，需为5个通道")
            return None, None
        
        # 读取红光波段（第4通道）和近红外波段（第5通道）
        red_band = dataset.GetRasterBand(4).ReadAsArray().astype(np.float32)
        nir_band = dataset.GetRasterBand(5).ReadAsArray().astype(np.float32)
        
        # 按论文标准进行归一化（除以各通道最大值）
        red_band /= self.normalization["max"][3]  # 红光波段最大值1267
        nir_band /= self.normalization["max"][4]  # 近红外波段最大值2612
        
        return red_band, nir_band
    
    def read_label(self, label_path):
        """读取标签数据"""
        dataset = gdal.Open(label_path)
        if dataset is None:
            print(f"无法打开标签: {label_path}")
            return None
        return dataset.ReadAsArray()
    
    def calculate_ndvi(self, red_band, nir_band):
        """计算NDVI，处理分母为零的情况"""
        denominator = nir_band + red_band
        mask = denominator != 0  # 避免除零错误
        
        ndvi = np.zeros_like(red_band, dtype=np.float32)
        ndvi[mask] = (nir_band[mask] - red_band[mask]) / denominator[mask]
        
        # 裁剪异常值（理论NDVI范围为[-1, 1]）
        ndvi = np.clip(ndvi, -1, 1)
        return ndvi
    
    def generate_file_lists(self, output_image_list='image_list.txt', output_label_list='label_list.txt'):
        """生成影像和标签路径列表文件"""
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(self.image_ext)])
        label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith(self.label_ext)])
        
        assert len(image_files) == len(label_files), "影像和标签数量不匹配"
        
        with open(output_image_list, 'w') as f:
            for img in image_files:
                f.write(os.path.join(self.image_dir, img) + '\n')
        
        with open(output_label_list, 'w') as f:
            for lbl in label_files:
                f.write(os.path.join(self.label_dir, lbl) + '\n')
        
        print(f"已生成影像列表: {output_image_list}")
        print(f"已生成标签列表: {output_label_list}")
        return output_image_list, output_label_list
    
    def analyze(self):
        """执行NDVI分析流程"""
        image_list, label_list = self.generate_file_lists()
        
        with open(image_list, 'r') as f:
            image_paths = [line.strip() for line in f]
        with open(label_list, 'r') as f:
            label_paths = [line.strip() for line in f]
        
        for image_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
            # 读取并预处理波段数据
            red_band, nir_band = self.read_image(image_path)
            label = self.read_label(label_path)
            
            if red_band is None or nir_band is None or label is None:
                continue
            
            # 计算NDVI
            ndvi = self.calculate_ndvi(red_band, nir_band)
            
            # 按类别收集NDVI值
            classes = np.unique(label)
            for cls in classes:
                if cls not in self.ndvi_by_class:
                    self.ndvi_by_class[cls] = []
                
                mask = label == cls
                self.ndvi_by_class[cls].extend(ndvi[mask].flatten().tolist())
        
        # 计算统计指标并确保所有值为Python原生类型
        statistics = {}
        for cls, ndvi_values in self.ndvi_by_class.items():
            if not ndvi_values:
                continue
                
            class_name = self.class_names[cls] if (self.class_names and cls < len(self.class_names)) else f"类别_{cls}"
            # 强制转换为Python原生类型（解决uint8等numpy类型问题）
            statistics[class_name] = {
                "类别ID": int(cls),  # 确保为int而非numpy整数类型
                "像素数量": int(len(ndvi_values)),
                "NDVI平均值": float(np.mean(ndvi_values)),
                "NDVI标准差": float(np.std(ndvi_values)),
                "NDVI最小值": float(np.min(ndvi_values)),
                "NDVI最大值": float(np.max(ndvi_values))
            }
        
        return statistics
    
    def visualize(self, stats, output_path=None):
        """可视化各类型NDVI分布，已配置中文字体"""
        class_names = list(stats.keys())
        mean_ndvi = [stats[name]["NDVI平均值"] for name in class_names]
        std_ndvi = [stats[name]["NDVI标准差"] for name in class_names]
        
        plt.figure(figsize=(12, 6))
        plt.bar(class_names, mean_ndvi, yerr=std_ndvi, capsize=5, color='skyblue')
        plt.axhline(y=0, color='r', linestyle='--', label='NDVI=0')
        plt.ylabel('NDVI值')
        plt.title('LCMA数据集各类别NDVI平均值与标准差')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存至: {output_path}")
        plt.show()
    
    def save_results(self, stats, output_dir):
        """保存统计结果至CSV和JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV
        df = pd.DataFrame.from_dict(stats, orient='index')
        df.to_csv(os.path.join(output_dir, 'ndvi_statistics.csv'))
        
        # 保存为JSON（已在analyze中确保所有值为原生类型）
        with open(os.path.join(output_dir, 'ndvi_statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        
        print(f"统计结果已保存至: {output_dir}")

if __name__ == "__main__":
    # 配置参数（根据实际路径修改）
    config = {
        "image_dir": "/data/ljt/Test/dataset/5bands",  # 影像目录
        "label_dir": "/data/ljt/Test/dataset/20label",  # 标签目录
        "output_dir": "ndvi_results",  # 结果输出目录
        "class_names": [
            "露天矿坑", "矿石加工场", "排土场", "水田", "蔬菜和水果温室",
            "绿色旱地", "灰色旱地", "苗圃和果园", "林地", "受胁迫的森林",
            "灌木林", "池塘和溪流", "矿坑池塘", "深色道路", "明亮道路",
            "浅灰色道路", "红色屋顶", "明亮屋顶", "蓝色屋顶", "裸地"
        ]  # 与论文TABLE I一致
    }
    
    # 执行分析
    analyzer = NDVIAnalyzer(
        config["image_dir"],
        config["label_dir"],
        config["class_names"]
    )
    
    statistics = analyzer.analyze()
    analyzer.save_results(statistics, config["output_dir"])
    analyzer.visualize(statistics, os.path.join(config["output_dir"], "ndvi_visualization.png"))