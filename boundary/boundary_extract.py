import cv2
import numpy as np

def extract_mask_edge(shadow_mask, desired_width=5):
    """
    用膨胀和腐蚀的差值提取固定宽度（约5像素）的阴影边界区域。
    """
    # 计算合适的核大小，需为奇数，确保宽度近似为 desired_width
    kernel_radius = max(1, desired_width // 2)
    kernel_size = (kernel_radius * 2 + 1, kernel_radius * 2 + 1)
    kernel = np.ones(kernel_size, np.uint8)

    dilated = cv2.dilate(shadow_mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(shadow_mask.astype(np.uint8), kernel, iterations=1)
    edge_mask = cv2.subtract(dilated, eroded)

    return edge_mask

def boundary_extract(imagepath, maskpath):
    # 读取图像
    rgb_image = cv2.imread(imagepath)
    shadow_mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)

    # 转换 BGR → RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # 二值化阴影掩码
    _, shadow_mask = cv2.threshold(shadow_mask, 127, 1, cv2.THRESH_BINARY)

    # 提取宽度约为 5 像素的边界
    edge_mask = extract_mask_edge(shadow_mask, desired_width=5)

    # 可视化叠加结果
    edge_vis = rgb_image.copy()
    edge_vis[edge_mask > 0] = [255, 0, 0]  # 标红边界

    # 显示结果
    # cv2.imshow("Shadow mask", shadow_mask * 255)
    # cv2.imshow("Edge mask (~5px)", edge_mask * 255)
    # cv2.imshow("Edge on image", cv2.cvtColor(edge_vis, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return edge_mask
if __name__ == '__main__':
    image_path = r"D:\Desktop\SRD\shadow\105.png"
    mask_path = r"D:\Desktop\SRD\mask\105.png"
    edge = boundary_extract(image_path, mask_path)
