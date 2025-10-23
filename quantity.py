'''
        ——SRI 用于衡量阴影消除后，阴影区域与无阴影区域在亮度或强度上的相似程度。
            值越接近 1，表示阴影区域恢复后与无阴影区域的亮度越接近，说明阴影消除效果越好
        ——CD 用于衡量阴影消除后，阴影区域与无阴影区域在颜色上的差异。
            值越小，表示阴影区域与无阴影区域的颜色越接近，说明阴影消除在颜色恢复方面效果越好。
        ——NIQE 是一种无参考图像质量评估指标，用于衡量图像的自然度，即图像是否看起来“真实”或“自然”。
            值越小，表示图像质量越高，越接近自然图像的统计特性。
            NIQE 特别适合评估阴影消除后图像的整体质量或阴影区域的质量。
'''
import numpy as np
from skimage import io, color, filters
from scipy.ndimage import label
import cv2
import os
from scipy import linalg


def load_image_and_mask(image_path, vmask_path, smask_path):
    """读取图像和两种掩码（vmask 和 smask）"""
    image = io.imread(image_path)
    vmask = io.imread(vmask_path)
    smask = cv2.imread(smask_path, cv2.IMREAD_GRAYSCALE)
    if image.shape[:2] != vmask.shape[:2] or image.shape[:2] != smask.shape:
        raise ValueError("Image, vmask, and smask dimensions must match")
    if len(image.shape) == 3 and image.shape[2] != 3:
        raise ValueError("Image must be RGB")
    if len(vmask.shape) == 3 and vmask.shape[2] != 3:
        raise ValueError("vmask must be RGB")
    return image, vmask, smask


def extract_region_pairs(vmask):
    """从 vmask（四色掩码）中提取阴影和无阴影区域对"""
    vmask_flat = np.dot(vmask[..., :3], [299, 587, 114]) / 1000  # 亮度加权
    unique_colors = np.unique(vmask_flat)

    region_pairs = []
    for color in unique_colors:
        binary_mask = (vmask_flat == color).astype(np.uint8)
        labeled, num_features = label(binary_mask)
        if num_features == 2:  # 假设每个颜色有正好两个区域（阴影和无阴影）
            areas = [(i, np.sum(labeled == i)) for i in range(1, num_features + 1)]
            areas.sort(key=lambda x: x[1], reverse=True)
            shadow_region = labeled == areas[0][0]
            non_shadow_region = labeled == areas[1][0]
            region_pairs.append((shadow_region, non_shadow_region))

    return region_pairs


def calculate_sri(image, shadow_mask, non_shadow_mask):
    """计算阴影恢复指数（SRI）"""
    gray_image = color.rgb2gray(image) if len(image.shape) == 3 else image
    mu_s = np.mean(gray_image[shadow_mask]) if np.any(shadow_mask) else 0
    mu_n = np.mean(gray_image[non_shadow_mask]) if np.any(non_shadow_mask) else 0

    if mu_s == 0 or mu_n == 0:
        return 0.0
    return 1 - abs(mu_s - mu_n) / max(mu_s, mu_n)


def calculate_cd(image, shadow_mask, non_shadow_mask):
    """计算色差（CD）"""
    lab_image = color.rgb2lab(image) if len(image.shape) == 3 else np.zeros((image.shape[0], image.shape[1], 3))
    mu_s = np.mean(lab_image[shadow_mask], axis=0) if np.any(shadow_mask) else np.zeros(3)
    mu_n = np.mean(lab_image[non_shadow_mask], axis=0) if np.any(non_shadow_mask) else np.zeros(3)

    delta_e = np.sqrt(np.sum((mu_s - mu_n) ** 2))
    return delta_e


def calculate_niqe_score(image_path, mode='a', smask_path=None, patch_size=7):
    """Calculate the NIQE (Natural Image Quality Evaluator) score"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image could not be loaded. Please check the file path.")

    img = img.astype(np.float32) / 255.0
    patches = []

    if mode == 's':
        if smask_path is None:
            raise ValueError("smask path must be provided when mode is 's'.")
        smask = cv2.imread(smask_path, cv2.IMREAD_GRAYSCALE)
        if smask is None:
            raise ValueError("smask could not be loaded. Please check the file path.")
        if smask.shape != img.shape:
            raise ValueError("smask dimensions do not match image dimensions.")

        for i in range(0, img.shape[0] - patch_size + 1, patch_size):
            for j in range(0, img.shape[1] - patch_size + 1, patch_size):
                if np.any(smask[i:i + patch_size, j:j + patch_size] == 255):
                    patch = img[i:i + patch_size, j:j + patch_size]
                    if patch.shape == (patch_size, patch_size):
                        patches.append(patch.flatten())
    elif mode == 'a':
        for i in range(0, img.shape[0] - patch_size + 1, patch_size):
            for j in range(0, img.shape[1] - patch_size + 1, patch_size):
                patch = img[i:i + patch_size, j:j + patch_size]
                if patch.shape == (patch_size, patch_size):
                    patches.append(patch.flatten())
    else:
        raise ValueError("Mode must be 'a' (all) or 's' (shadow).")

    if len(patches) == 0:
        raise ValueError("No valid patches found for NIQE calculation.")

    features = []
    for patch in patches:
        mean = np.mean(patch)
        variance = np.var(patch)
        cov_like = np.mean(np.abs(patch - mean))
        features.append([mean, variance, cov_like])

    features = np.array(features)
    mu_pris = np.array([0.5, 0.1, 0.05])
    cov_pris = np.array([
        [0.01, 0.001, 0.0001],
        [0.001, 0.01, 0.001],
        [0.0001, 0.001, 0.01]
    ])
    mu_dist = np.mean(features, axis=0)
    cov_dist = np.cov(features.T)
    mu_diff = mu_dist - mu_pris
    inv_cov_pris = linalg.inv(cov_pris)
    niqe_score = np.sqrt(mu_diff.T @ inv_cov_pris @ mu_diff)

    return niqe_score


def calculate_shadow_metrics(image_path, vmask_path, smask_path):
    """主函数：计算SRI、CD、NIQE_Score_All、NIQE_Score_Shadow"""
    image, vmask, smask = load_image_and_mask(image_path, vmask_path, smask_path)
    region_pairs = extract_region_pairs(vmask)

    sri_values, cd_values = [], []

    for shadow_mask, non_shadow_mask in region_pairs:
        sri = calculate_sri(image, shadow_mask, non_shadow_mask)
        cd = calculate_cd(image, shadow_mask, non_shadow_mask)
        sri_values.append(sri)
        cd_values.append(cd)

    niqe_score_all = calculate_niqe_score(image_path, mode='a')
    niqe_score_shadow = calculate_niqe_score(image_path, mode='s', smask_path=smask_path)

    return {
        'SRI': np.mean(sri_values) if sri_values else float('nan'),
        'CD': np.mean(cd_values) if cd_values else float('nan'),
        'NIQE_Score_All': niqe_score_all,
        'NIQE_Score_Shadow': niqe_score_shadow
    }


def process_folders(image_folder, vmask_folder, smask_folder, output_file="shadowmetrics.txt"):
    """处理文件夹中的图像和掩码，写入评估结果"""
    all_sri, all_cd, all_niqe_a, all_niqe_s = [], [], [], []

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Image_Name,SRI,CD,NIQE_Score_All,NIQE_Score_Shadow\n")

        for image_file in os.listdir(image_folder):
            if image_file.endswith(('.tif', '.png')):
                base_name = image_file.replace('_deshadowed', '').rsplit('.', 1)[0]
                vmask_file = base_name + '.png'  # 假设 vmask 文件为 .png
                smask_file = base_name + '.tif'  # 假设 smask 文件为 .png
                image_path = os.path.join(image_folder, image_file)
                vmask_path = os.path.join(vmask_folder, vmask_file)
                smask_path = os.path.join(smask_folder, smask_file)

                if os.path.exists(vmask_path) and os.path.exists(smask_path):
                    try:
                        metrics = calculate_shadow_metrics(image_path, vmask_path, smask_path)
                        sri = metrics['SRI']
                        cd = metrics['CD']
                        niqe_a = metrics['NIQE_Score_All']
                        niqe_s = metrics['NIQE_Score_Shadow']
                        f.write(f"{image_file},{sri:.4f},{cd:.4f},{niqe_a:.4f},{niqe_s:.4f}\n")
                        all_sri.append(sri)
                        all_cd.append(cd)
                        all_niqe_a.append(niqe_a)
                        all_niqe_s.append(niqe_s)
                        print(
                            f"Processed {image_file}: SRI={sri:.4f}, CD={cd:.4f}, NIQE_Score_All={niqe_a:.4f}, NIQE_Score_Shadow={niqe_s:.4f}")
                    except Exception as e:
                        f.write(
                            f"{image_file},Error: {str(e)}. Please install 'imagecodecs' package if using LZW-compressed TIFF files.\n")
                else:
                    f.write(f"{image_file},Mask Not Found\n")

        avg_sri = np.mean(all_sri) if all_sri else float('nan')
        avg_cd = np.mean(all_cd) if all_cd else float('nan')
        avg_niqe_a = np.mean(all_niqe_a) if all_niqe_a else float('nan')
        avg_niqe_s = np.mean(all_niqe_s) if all_niqe_s else float('nan')
        f.write(f"Average,{avg_sri:.4f},{avg_cd:.4f},{avg_niqe_a:.4f},{avg_niqe_s:.4f}\n")


if __name__ == "__main__":
    image_folder = r"D:\Desktop\CompareExp\AISD\RSiSDSR\gtMSRresult"  # 图像文件夹
    vmask_folder = r"AISD/vmask"  # vmask 文件夹（四色掩码）
    smask_folder = r"AISD/smask"  # smask 文件夹（二值掩码）
    output_file = r"AISD/RSiSDSRmetrics.txt"
    process_folders(image_folder, vmask_folder, smask_folder, output_file)
