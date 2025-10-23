import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import measure, graph
import networkx as nx
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from skimage.color import rgb2lab
from skimage.feature import local_binary_pattern
import warnings

# Suppress specific skimage warnings
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern` to floating-point images")
warnings.filterwarnings("ignore", message="y_pred has been converted to an array of type np.uint8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Import models and datasets from their respective files.
# Make sure these are accessible or copy-paste their definitions here.
# For simplicity, I'm assuming the model definitions are available.
# You might need to adjust these imports based on your exact file structure.
from utils.DBSCF_DCENet import ShadowDetectionNetwork
from boundary.boundary_smooth import boundary_smooth  # Assuming this module is in a 'boundary' folder

# --- Configuration Parameters for Shadow Detection ---
# Adjust these paths as per your environment
SHADOW_DETECTION_CHECKPOINT_PATH = r"D:\Desktop\SARU-Net\bestcheckpoint\best_AISD_ckp.pth"  # Update this path
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Common Image Processing Utilities ---
def imread_rgb(image_path):
    """Reads an image and converts it to RGB."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def imwrite_rgb(image, path):
    """Saves an RGB image."""
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Image saved to: {path}")


def imread_mask(mask_path):
    """Reads a mask and binarizes it."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot open mask: {mask_path}")
    _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
    return mask


# --- Shadow Detection Module (Adapted from test.py) ---
class ShadowDetector:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = ShadowDetectionNetwork().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    def detect_shadow_mask(self, image_path):
        """
        Detects shadow mask for a single input image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            np.ndarray: Predicted binary shadow mask (255 for shadow, 0 for non-shadow), resized to original image dimensions.
            PIL.Image: Original image as PIL Image.
        """
        original_image_pil = Image.open(image_path).convert("RGB")
        original_width, original_height = original_image_pil.size

        input_image_tensor = self.transform(original_image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_mask_tensor = self.model(input_image_tensor)

        pred_mask_numpy = pred_mask_tensor.squeeze().cpu().numpy()
        pred_mask_binary = (pred_mask_numpy > 0.5).astype(np.uint8) * 255  # Convert to 0/255

        # Resize the predicted mask back to the original image dimensions
        pred_mask_resized = Image.fromarray(pred_mask_binary).resize(
            (original_width, original_height), Image.Resampling.NEAREST
        )
        return np.array(pred_mask_resized), original_image_pil



# --- Shadow Removal Module (Adapted from bath_sr.py) ---

def compute_bhattacharyya_coefficient(hist1, hist2):
    return np.sum(np.sqrt(hist1 * hist2))


def compute_histogram(pixels_flat, channel, bins=32):
    if pixels_flat.size == 0:
        return np.zeros(bins, dtype=np.float32)
    # Ensure pixels_flat is 1D for histogram, even if it has a channel dimension of 1
    channel_values = pixels_flat if pixels_flat.ndim == 1 else pixels_flat[:, channel]
    hist, _ = np.histogram(channel_values, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (np.sum(hist) + 1e-6)
    return hist


def global_illumination_correction(image_np, shadow_mask_binary):
    """
    Applies global illumination correction to the shadow regions.
    Args:
        image_np (np.ndarray): Original RGB image (uint8).
        shadow_mask_binary (np.ndarray): Binary mask (1 for shadow, 0 for non-shadow, uint8).
    Returns:
        np.ndarray: Corrected RGB image (uint8).
    """
    global_ratios = []
    un_shaded_mask = np.uint8(shadow_mask_binary == 0)  # 0 for shadow, 1 for non-shadow

    for i in range(3):  # For R, G, B channels
        # Ensure we are operating on the correct data types and values (0-255)
        shaded_region_pixels = image_np[:, :, i][shadow_mask_binary > 0]
        un_shaded_region_pixels = image_np[:, :, i][un_shaded_mask > 0]

        shaded_illumination = np.mean(shaded_region_pixels) if shaded_region_pixels.size > 0 else 1e-6
        un_shaded_illumination = np.mean(un_shaded_region_pixels) if un_shaded_region_pixels.size > 0 else 1e-6

        global_ratio = un_shaded_illumination / shaded_illumination
        global_ratios.append(global_ratio)

    corrected_image = image_np.copy()
    shadow_area = np.expand_dims(shadow_mask_binary, axis=2)  # 2D to 3D
    global_ratios = np.array(global_ratios, dtype=np.float32)
    global_ratios_tiled = np.tile(global_ratios.reshape(1, 1, 3), (image_np.shape[0], image_np.shape[1], 1))

    # Apply correction only to shadow regions
    corrected_image = np.where(
        shadow_area > 0,  # If it's a shadow pixel
        np.clip(image_np * global_ratios_tiled, 0, 255).astype(np.uint8),  # Apply ratio
        image_np  # Keep original if not shadow
    )
    return corrected_image


def segment_shadow_and_nonshadow(image_np, shadow_mask_binary, pixel=600, compactness=20, sigma=1):
    """
    Performs SLIC superpixel segmentation separately for shadow and non-shadow regions.
    Parameters:
        image_np: np.ndarray Original RGB image (uint8)
        shadow_mask_binary: np.ndarray Binary mask (values 0 or 1, uint8)
        pixel: Helps control the number of superpixels
        compactness: Compactness parameter, higher value means more regular superpixels
        sigma: Smoothing parameter
    Returns:
        merged_labels: Merged superpixel label map (labels start from 1)
    """
    image_float = image_np.astype(np.float32) / 255.0  # Convert to float [0, 1] for slic

    # Shadow and non-shadow masks
    shadow_area = (shadow_mask_binary > 0)
    non_shadow_area = (shadow_mask_binary == 0)

    # Segment shadow regions separately
    shadow_labels = np.zeros_like(shadow_mask_binary, dtype=np.int32)
    if np.any(shadow_area):
        num_segments_shadow = int(np.sum(shadow_area) / pixel)
        if num_segments_shadow == 0: num_segments_shadow = 1  # Ensure at least one segment
        shadow_slic = slic(image_float, n_segments=num_segments_shadow,
                           compactness=compactness, sigma=sigma, start_label=1, mask=shadow_area)
        shadow_labels[shadow_area] = shadow_slic[shadow_area]

    # Segment non-shadow regions separately
    non_shadow_labels = np.zeros_like(shadow_mask_binary, dtype=np.int32)
    if np.any(non_shadow_area):
        num_segments_non_shadow = int(np.sum(non_shadow_area) / pixel)
        if num_segments_non_shadow == 0: num_segments_non_shadow = 1  # Ensure at least one segment
        offset = shadow_labels.max() if np.any(shadow_labels) else 0  # Start non-shadow labels after shadow labels
        non_shadow_slic = slic(image_float, n_segments=num_segments_non_shadow,
                               compactness=compactness, sigma=sigma, start_label=1, mask=non_shadow_area)
        non_shadow_labels[non_shadow_area] = non_shadow_slic[non_shadow_area] + offset

    # Merge label maps
    merged_labels = shadow_labels + non_shadow_labels
    return merged_labels


def shadow_removal_core(image_np, shadow_mask_binary, temp_output_dir):
    """
    Performs the core shadow removal logic.
    Args:
        image_np (np.ndarray): Original RGB image (uint8).
        shadow_mask_binary (np.ndarray): Binary mask (1 for shadow, 0 for non-shadow, uint8).
        temp_output_dir (str): Temporary directory to save intermediate files needed by boundary_smooth.
    Returns:
        np.ndarray: Deshadowed RGB image (uint8).
    """
    print("Starting shadow removal process...")

    # Step 1: Preliminary optimization using global illumination information
    image_pre_corrected = global_illumination_correction(image_np, shadow_mask_binary)
    print("Global illumination correction completed.")

    # Step 2: Superpixel segmentation
    segment_labels = segment_shadow_and_nonshadow(image_pre_corrected, shadow_mask_binary, compactness=10, sigma=1)
    print("Superpixel segmentation completed.")
    # Optional: Visualization - you can uncomment this if you want to see superpixel boundaries
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mark_boundaries(image_pre_corrected, segment_labels))
    # plt.title("Superpixel Boundaries")
    # plt.axis('off')
    # plt.show()

    # Calculate mask region properties
    regionsmask = measure.regionprops(segment_labels, intensity_image=shadow_mask_binary)
    region_is_shadows = np.array(
        [r.mean_intensity > 0.5 for r in regionsmask])  # Check if mean_intensity > 0 for binary mask

    # Recalculate global illumination ratios for limiting exposure later
    global_ratios = []
    un_shaded_mask_for_ratio = np.uint8(shadow_mask_binary == 0)  # 0 for shadow, 1 for non-shadow
    for i in range(3):
        shaded_region_pixels = image_pre_corrected[:, :, i][shadow_mask_binary > 0]
        un_shaded_region_pixels = image_pre_corrected[:, :, i][un_shaded_mask_for_ratio > 0]
        shaded_illumination = np.mean(shaded_region_pixels) if shaded_region_pixels.size > 0 else 1e-6
        un_shaded_illumination = np.mean(un_shaded_region_pixels) if un_shaded_region_pixels.size > 0 else 1e-6
        global_ratio = un_shaded_illumination / shaded_illumination
        global_ratios.append(global_ratio)

    # Calculate image region properties
    regionsimage = measure.regionprops(segment_labels, intensity_image=image_pre_corrected)
    region_intensity_images = [r.intensity_image for r in regionsimage]  # RGB pixel values for each superpixel

    # Calculate centroids for all superpixels
    regionsxy = measure.regionprops(segment_labels)
    centroids = np.array([r.centroid for r in regionsxy])

    numbers = 5  # Number of nearest non-shadow superpixels

    corrected_img = image_pre_corrected * np.expand_dims(un_shaded_mask_for_ratio,
                                                         axis=2)  # Initialize with non-shadow areas

    # Process each shadow superpixel
    region_shadow_indexs = np.where(region_is_shadows)[0]
    for region_shadow_index in tqdm(region_shadow_indexs, desc="Optimizing shadow superpixels"):
        region_shadow_intensity_image = region_intensity_images[region_shadow_index]

        # Ensure the superpixel region is valid before processing
        if region_shadow_intensity_image is None or region_shadow_intensity_image.size == 0 or np.sum(
                region_shadow_intensity_image) == 0:
            continue

        shaded_illumination_bands = []
        for i in range(3):
            valid_pixels = region_shadow_intensity_image[:, :, i][region_shadow_intensity_image[:, :, i] > 0]
            shaded_illumination_bands.append(np.mean(valid_pixels) if valid_pixels.size > 0 else 1e-6)

        shadow_centroid = centroids[region_shadow_index]
        non_shadow_indices = np.where(~region_is_shadows)[0]

        if len(non_shadow_indices) == 0:
            # Fallback if no non-shadow regions exist (e.g., image is entirely shadow)
            ratio_bands = np.array(global_ratios)
        else:
            distances = np.sqrt(np.sum((centroids[non_shadow_indices] - shadow_centroid) ** 2, axis=1))
            nearest_indices = non_shadow_indices[np.argsort(distances)[:min(numbers, len(non_shadow_indices))]]
            expanded_adj_indexs = set(nearest_indices.tolist())

            alpha = 0.6  # EMD weight
            beta = 0.3  # LBP Bhattacharyya distance weight
            gamma = 0.1  # a-channel mean difference weight

            shadow_lab_pixels = rgb2lab(region_shadow_intensity_image.astype(np.float32) / 255.0)
            valid_pixel_mask_shadow = (region_shadow_intensity_image[:, :, 0] > 0)
            shadow_lab_flat = shadow_lab_pixels[valid_pixel_mask_shadow].reshape(-1, 3)

            shadow_gray = np.mean(region_shadow_intensity_image, axis=2)
            shadow_lbp = local_binary_pattern(shadow_gray.astype(np.uint8), P=8, R=1, method='uniform')
            shadow_lbp_flat = shadow_lbp[valid_pixel_mask_shadow].ravel()

            shadow_a_mean = np.mean(shadow_lab_flat[:, 1]) if shadow_lab_flat.shape[0] > 0 else 0

            similarities = []
            un_shaded_illuminations = []

            for region_shadow_adj_index in expanded_adj_indexs:
                if not region_is_shadows[region_shadow_adj_index]:
                    region_intensity_image = region_intensity_images[region_shadow_adj_index]
                    if region_intensity_image is None or region_intensity_image.size == 0 or np.sum(
                            region_intensity_image) == 0:
                        continue

                    region_lab_pixels = rgb2lab(region_intensity_image.astype(np.float32) / 255.0)
                    valid_pixel_mask_non_shadow = (region_intensity_image[:, :, 0] > 0)
                    region_lab_flat = region_lab_pixels[valid_pixel_mask_non_shadow].reshape(-1, 3)

                    region_gray = np.mean(region_intensity_image, axis=2)
                    region_lbp = local_binary_pattern(region_gray.astype(np.uint8), P=8, R=1, method='uniform')
                    region_lbp_flat = region_lbp[valid_pixel_mask_non_shadow].ravel()

                    emd = 0.0
                    if shadow_lab_flat.shape[0] > 0 and region_lab_flat.shape[0] > 0:
                        d_L = wasserstein_distance(shadow_lab_flat[:, 0], region_lab_flat[:, 0])
                        d_a = wasserstein_distance(shadow_lab_flat[:, 1], region_lab_flat[:, 1])
                        d_b = wasserstein_distance(shadow_lab_flat[:, 2], region_lab_flat[:, 2])
                        emd = (d_L + d_a + d_b) / 3.0

                    lbp_bhat = 0.0
                    if shadow_lbp_flat.shape[0] > 0 and region_lbp_flat.shape[0] > 0:
                        hist1 = compute_histogram(shadow_lbp_flat.reshape(-1, 1), channel=0, bins=10)
                        hist2 = compute_histogram(region_lbp_flat.reshape(-1, 1), channel=0, bins=10)
                        lbp_bhat = compute_bhattacharyya_coefficient(hist1, hist2)

                    region_a_mean = np.mean(region_lab_flat[:, 1]) if region_lab_flat.shape[0] > 0 else 0
                    a_mean_diff = abs(shadow_a_mean - region_a_mean)

                    emd_norm = emd / 100.0
                    lbp_bhat_norm = 1.0 - lbp_bhat
                    a_mean_diff_norm = a_mean_diff / 128.0

                    distance_combined = alpha * emd_norm + beta * lbp_bhat_norm + gamma * a_mean_diff_norm
                    similarity = 1.0 / (distance_combined + 1e-6)
                    similarities.append(similarity)

                    illumination_bands = []
                    for i in range(3):
                        valid_pixels = region_intensity_image[:, :, i][region_intensity_image[:, :, i] > 0]
                        illumination_bands.append(np.mean(valid_pixels) if valid_pixels.size > 0 else 1e-6)
                    un_shaded_illuminations.append(illumination_bands)

            similarities = np.array(similarities)
            if len(similarities) > 0 and np.sum(similarities) > 1e-6:
                weights = similarities / np.sum(similarities)
                un_shaded_illumination_bands = np.average(np.array(un_shaded_illuminations), axis=0, weights=weights)
                shaded_illumination_bands = np.array(shaded_illumination_bands)
                ratio_bands = un_shaded_illumination_bands / (shaded_illumination_bands + 1e-6)
            else:
                ratio_bands = np.array(global_ratios)

        for i in range(3):
            if ratio_bands[i] > global_ratios[i]:
                ratio_bands[i] = global_ratios[i]
            if ratio_bands[i] < 1:
                ratio_bands[i] = 1

        current_region_mask = (segment_labels == (region_shadow_index + 1))
        if not np.any(current_region_mask):
            continue

        image_region = image_pre_corrected * np.expand_dims(current_region_mask, axis=2)
        ratio_bands_tiled = np.tile(ratio_bands.reshape(1, 1, -1), (image_np.shape[0], image_np.shape[1], 1))
        image_region_correct = np.clip(image_region * ratio_bands_tiled, 0, 255).astype(np.uint8)
        corrected_img = np.where(np.expand_dims(current_region_mask, axis=2), image_region_correct, corrected_img)

    print("Shadow pixel value adjustment completed.")

    # Step 3: Boundary optimization
    # boundary_smooth function expects file paths. We'll save the intermediate corrected_img.
    temp_corrected_path = os.path.join(temp_output_dir, "temp_corrected.png")
    temp_shadow_mask_path = os.path.join(temp_output_dir, "temp_mask.png")
    imwrite_rgb(corrected_img, temp_corrected_path)
    cv2.imwrite(temp_shadow_mask_path, shadow_mask_binary * 255)  # Save mask as 0/255 for boundary_smooth

    final_image = boundary_smooth(temp_corrected_path, temp_shadow_mask_path)
    print("Shadow boundary smoothing completed.")

    # Clean up temporary files
    os.remove(temp_corrected_path)
    os.remove(temp_shadow_mask_path)

    return final_image


# --- Main Demo Function ---
def run_shadow_demo(input_image_path, output_root_dir):
    """
    Runs the full shadow detection and removal pipeline for a single image.
    Args:
        input_image_path (str): Path to the input image with shadows.
        output_root_dir (str): Root directory to save all results.
    Returns:
        tuple: (deshadowed_image_np, predicted_mask_np)
    """
    # Create output directories
    demo_output_dir = os.path.join(output_root_dir, "demoresult")
    detection_output_dir = os.path.join(demo_output_dir, "detected_masks")
    removal_output_dir = os.path.join(demo_output_dir, "removed_shadows")
    temp_dir = os.path.join(demo_output_dir, "temp")  # For intermediate files

    os.makedirs(detection_output_dir, exist_ok=True)
    os.makedirs(removal_output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    original_ext = os.path.splitext(os.path.basename(input_image_path))[1]

    print(f"--- Processing {input_image_path} ---")

    # 1. Shadow Detection
    print("\n--- Step 1: Shadow Detection ---")
    shadow_detector = ShadowDetector(SHADOW_DETECTION_CHECKPOINT_PATH, DEVICE)
    predicted_mask_np, original_image_pil = shadow_detector.detect_shadow_mask(input_image_path)

    # Save detected mask
    detected_mask_path = os.path.join(detection_output_dir, f"{base_name}_detected_mask{original_ext}")
    cv2.imwrite(detected_mask_path, predicted_mask_np)
    print(f"Detected shadow mask saved to: {detected_mask_path}")

    # Save detection canvas
    canvas_path = os.path.join(detection_output_dir, f"{base_name}_detection_canvas.png")

    # 2. Shadow Removal
    print("\n--- Step 2: Shadow Removal ---")
    original_image_np = imread_rgb(input_image_path)
    # Convert predicted_mask_np (0/255) to binary (0/1) for shadow removal core logic
    binary_shadow_mask = (predicted_mask_np > 127).astype(np.uint8)

    deshadowed_image_np = shadow_removal_core(original_image_np, binary_shadow_mask, temp_dir)

    # Save deshadowed image
    deshadowed_image_path = os.path.join(removal_output_dir, f"{base_name}_deshadowed{original_ext}")
    imwrite_rgb(deshadowed_image_np, deshadowed_image_path)
    print(f"Deshadowed image saved to: {deshadowed_image_path}")

    print(f"\n--- Demo for {base_name} Completed Successfully! ---")
    print(f"Results are in: {demo_output_dir}")

    # Optional: Display images (requires matplotlib)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image_pil)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask_np, cmap='gray')
    plt.title("Predicted Shadow Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(deshadowed_image_np)
    plt.title("Deshadowed Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Clean up the temporary directory if it's empty, or remove its contents
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    elif os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)

    return deshadowed_image_np, predicted_mask_np


if __name__ == "__main__":
    # --- Example Usage ---
    # !!! IMPORTANT: Update this path to your actual image file !!!
    # For testing, you can use an image like:
    # input_single_image_path = r"D:\BaiduNetdiskDownload\AISD\Test51\shadow\001.png"
    input_single_image_path = r"D:\Desktop\SARU-Net\demo\JiangXi_54.png"  # Replace with your test image path
    output_root_directory = r""  # Results will be in output_root_directory/demoresult

    if not os.path.exists(input_single_image_path):
        print(f"Error: Input image not found at {input_single_image_path}")
        print("Please update 'input_single_image_path' in demo.py to a valid image file.")
    else:
        run_shadow_demo(input_single_image_path, output_root_directory)