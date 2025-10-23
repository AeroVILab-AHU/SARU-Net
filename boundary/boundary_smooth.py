import cv2
import numpy as np
import os
from .boundary_extract import boundary_extract


def preprocess_shadow_region(image, shadow_mask, edge_mask, ks=3):
    """
    预处理边界区域内的阴影像素，替换为无阴影区域的平均值。

    参数:
        image (np.ndarray): RGB 图像
        shadow_mask (np.ndarray): 阴影掩码（0 表示无阴影，1 表示阴影）
        edge_mask (np.ndarray): 边界掩码（255 表示边界像素）
        ks (int): 邻域大小（奇数）

    返回:
        preprocessed_image (np.ndarray): 预处理后的图像
    """
    preprocessed_image = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    center = ks // 2

    boundary_pixels = np.where(edge_mask == 255)
    for x, y in zip(boundary_pixels[0], boundary_pixels[1]):
        if shadow_mask[x, y] == 1:
            rgb_values = []
            for i in range(-center, center + 1):
                for j in range(-center, center + 1):
                    ni, nj = x + i, y + j
                    if 0 <= ni < h and 0 <= nj < w and shadow_mask[ni, nj] == 0:
                        rgb_values.append(image[ni, nj].copy())
            if rgb_values:
                preprocessed_image[x, y] = np.mean(rgb_values, axis=0)

    return preprocessed_image


def boundary_smooth(input_image_path, shadow_mask_path, output_path='', d=3, sigmaColor=75, sigmaSpace=75, ksize=(3, 3),
                    sigma=1.0, iterations=1, dilate_ks=3):
    """
    使用双边滤波器和高斯滤波器依次平滑阴影边界区域，保存中间图像。

    参数:
        input_image_path (str): 输入 RGB 图像路径
        shadow_mask_path (str): 阴影掩码路径（灰度图）
        output_path (str): 输出平滑后图像的保存路径
        d (int): 双边滤波邻域直径
        sigmaColor (float): 双边滤波颜色标准差
        sigmaSpace (float): 双边滤波空间标准差
        ksize (tuple): 高斯滤波核大小（奇数，例如 (5, 5)）
        sigma (float): 高斯滤波标准差
        iterations (int): 滤波迭代次数
        dilate_ks (int): 边界膨胀核大小（奇数）
    """
    # 读取图像
    rgb_image = cv2.imread(input_image_path)
    if rgb_image is None:
        raise ValueError(f"无法读取图像: {input_image_path}")

    shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
    if shadow_mask is None:
        raise ValueError(f"无法读取阴影掩码: {shadow_mask_path}")

    # 转换为 RGB 和确保 shadow_mask 为二值化 (0/1)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    _, shadow_mask = cv2.threshold(shadow_mask, 127, 1, cv2.THRESH_BINARY)

    # 调用 boundary_extract 获取边界掩码
    edge_mask = boundary_extract(input_image_path, shadow_mask_path)
    edge_mask = (edge_mask * 255).astype(np.uint8)

    # 检查 edge_mask 是否为空
    # if not np.any(edge_mask):
    #     print("警告: edge_mask 为空，未检测到边界区域，直接保存原始图像")
    #     cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    #     return

    # 扩展边界掩码
    kernel = np.ones((dilate_ks, dilate_ks), np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

    # 创建输出目录
    # output_dir = os.path.dirname(output_path)
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)

    # 保存原始图像和掩码
    # cv2.imwrite(os.path.join(output_dir, "input.png"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(output_dir, "edge_mask.png"), edge_mask)
    # cv2.imwrite(os.path.join(output_dir, "shadow_mask.png"), shadow_mask * 255)
    # print(f"原始图像和掩码已保存至: {output_dir}")

    # 预处理阴影区域
    preprocessed_image = preprocess_shadow_region(rgb_image, shadow_mask, edge_mask, ks=d)

    # 创建结果图像
    result = preprocessed_image.copy()

    # 迭代双边滤波
    for _ in range(iterations):
        result = cv2.bilateralFilter(
            result.astype(np.uint8),
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace
        ).astype(np.float32)

    # 融合双边滤波结果
    smoothed_bilateral = result.copy()
    for channel in range(3):
        smoothed_bilateral[:, :, channel] = np.where(
            edge_mask == 255,
            result[:, :, channel],
            preprocessed_image[:, :, channel]
        )

    # 保存双边滤波结果
    # cv2.imwrite(os.path.join(output_dir, "smoothed_bilateral.png"),
    #             cv2.cvtColor(smoothed_bilateral.astype(np.uint8), cv2.COLOR_RGB2BGR))
    # print(f"双边滤波图像已保存至: {os.path.join(output_dir, 'smoothed_bilateral.png')}")

    # 应用高斯滤波
    for _ in range(iterations):
        result = cv2.GaussianBlur(
            smoothed_bilateral,
            ksize=ksize,
            sigmaX=sigma,
            sigmaY=sigma
        ).astype(np.float32)

    # 融合高斯滤波结果
    for channel in range(3):
        result[:, :, channel] = np.where(
            edge_mask == 255,
            result[:, :, channel],
            preprocessed_image[:, :, channel]
        )

    # 转换为 uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 保存最终结果
    # cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    # print(f"最终平滑图像已保存至: {output_path}")
    return result


# 示例调用
if __name__ == "__main__":
    input_image_path = r'D:\Desktop\SRD\shadow\105__g2l_nb_deshadowed.png'
    shadow_mask_path = r'D:\Desktop\SRD\mask\105.png'
    output_path = r'D:\Desktop\SRD\shadow\105_dbsmoothed.png'
    # 设置参数
    d = 5  # 双边滤波邻域直径
    sigmaColor = 75  # 双边滤波颜色标准差
    sigmaSpace = 75  # 双边滤波空间标准差
    ksize = (5, 5)  # 高斯滤波核大小
    sigma = 1.0  # 高斯滤波标准差
    iterations = 1  # 迭代次数
    dilate_ks = 5  # 边界膨胀核大小
    boundary_smooth(input_image_path, shadow_mask_path, output_path, d, sigmaColor, sigmaSpace, ksize, sigma,
                    iterations, dilate_ks)