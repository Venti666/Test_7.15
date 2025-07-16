import os
import sys

def get_tif_info(file_path):
    """获取TIF文件的通道数和波段信息"""
    info = {
        "file_path": file_path,
        "channel_count": 0,
        "band_names": [],
        "driver": "",
        "width": 0,
        "height": 0,
        "dtype": "",
        "error": None
    }

    # 优先使用rasterio（推荐用于遥感数据）
    try:
        import rasterio
        with rasterio.open(file_path) as src:
            info["channel_count"] = src.count
            info["width"] = src.width
            info["height"] = src.height
            info["dtype"] = src.dtypes[0] if src.count > 0 else ""
            info["driver"] = "rasterio"
            
            # 尝试获取波段名称
            for i in range(1, src.count + 1):
                desc = src.descriptions[i-1] or f"Band {i}"
                info["band_names"].append(desc)
            
            # 检查是否有NDVI等特殊波段
            if src.count >= 2:
                try:
                    red = src.read(1).astype(float)
                    nir = src.read(2).astype(float)
                    ndvi = (nir - red) / (nir + red + 1e-10)
                    info["has_ndvi_potential"] = True
                except:
                    info["has_ndvi_potential"] = False
            return info
    
    except (ImportError, Exception) as e:
        pass

    # 备选：使用GDAL
    try:
        from osgeo import gdal
        ds = gdal.Open(file_path)
        if ds is None:
            info["error"] = "GDAL无法打开文件"
            return info
            
        info["channel_count"] = ds.RasterCount
        info["width"] = ds.RasterXSize
        info["height"] = ds.RasterYSize
        info["driver"] = "GDAL"
        
        # 获取数据类型
        band = ds.GetRasterBand(1)
        if band:
            dtype = gdal.GetDataTypeName(band.DataType)
            info["dtype"] = dtype
            
            # 尝试获取波段名称
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                desc = band.GetDescription() or f"Band {i}"
                info["band_names"].append(desc)
        
        ds = None  # 释放资源
        return info
    
    except (ImportError, Exception) as e:
        pass

    # 备选：使用PIL（仅适用于简单图像）
    try:
        from PIL import Image
        img = Image.open(file_path)
        info["channel_count"] = len(img.getbands())
        info["width"], info["height"] = img.size
        info["dtype"] = str(img.mode)
        info["driver"] = "PIL"
        info["band_names"] = list(img.getbands())
        return info
    
    except (ImportError, Exception) as e:
        info["error"] = "无法读取文件：未找到合适的库（需要rasterio、GDAL或PIL）"
        return info

def print_tif_info(info):
    """格式化输出TIF文件信息"""
    print(f"文件: {os.path.basename(info['file_path'])}")
    print(f"尺寸: {info['width']} x {info['height']} 像素")
    print(f"通道数: {info['channel_count']}")
    
    if info['band_names']:
        print("波段信息:")
        for i, name in enumerate(info['band_names'], 1):
            print(f"  波段 {i}: {name}")
    
    print(f"数据类型: {info['dtype']}")
    print(f"使用库: {info['driver']}")
    
    if info['error']:
        print(f"警告: {info['error']}")
    
    if 'has_ndvi_potential' in info:
        if info['has_ndvi_potential']:
            print("提示: 该TIF可能包含计算NDVI所需的红光和近红外波段")
        else:
            print("提示: 该TIF波段数不足，可能无法计算NDVI")

if __name__ == "__main__":
    # 使用方法: python tif_channel_checker.py your_file.tif
    if len(sys.argv) < 2:
        print("用法: python tif_channel_checker.py <TIF文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        sys.exit(1)
    
    info = get_tif_info(file_path)
    print_tif_info(info)