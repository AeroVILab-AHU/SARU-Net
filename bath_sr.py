# -*- coding: utf-8 -*-
"""
Physical-based shadow removal for remote sensing images (PNG format)
Code Description:
First, perform preliminary shadow removal by globally comparing illumination ratios in shadow regions.
Then, conduct superpixel segmentation separately for shadow and non-shadow regions.
Calculate global illumination ratios (maximum ratio).
Obtain the n nearest neighboring non-shadow superpixels for each shadow superpixel.
For each non-shadow superpixel, utilize EMD, LBP_Bhattacharyya, and a-Diff.
Further optimize shadow removal for each shadow superpixel using a linear adjustment method.
Finally, optimize the results with a boundary optimization algorithm.
"""
import glob
from skimage.util import img_as_float
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from skimage.segmentation import slic, find_boundaries, mark_boundaries
from skimage import measure, graph
import networkx as nx
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from skimage.color import rgb2lab
import warnings
from boundary.boundary_smooth import boundary_smooth # Assuming this module is in a 'boundary' folder

warnings.filterwarnings("ignore", message="Applying `local_binary_pattern` to floating-point images")


def get_deshadowed_path(output_dir, base_name, ext):
    return os.path.join(output_dir, f"{base_name}_deshadowed{ext}")

def get_smooth_path(output_dir, base_name, ext):
    # This function is kept, but its return value will no longer be used to save intermediate files.
    return os.path.join(output_dir, f"{base_name}_smoothed{ext}")

# Image I/O functions
def imread(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def imread_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot open mask: {mask_path}")
    # Binarize mask (threshold at 128)
    _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
    return mask

def imwrite(image, path):
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)
    print(f"Image saved to: {path}")

def plot_segmentation(image, segment_labels):
    # Visualization: Draw superpixel boundaries on the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    # Mark superpixel boundaries on the original shadow image
    boundary_image = mark_boundaries(image, segment_labels, color=(1, 0, 0))  # Red boundaries
    ax.imshow(boundary_image)
    ax.set_title("Superpixel Boundaries on Shadow Image")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_intensity_map(region_intensity_images, region_areas, n):
    cols = n // 5  # Number of images per row
    rows = 5      # Total of 5 rows
    plt.figure(figsize=(cols * 2, rows * 2))  # Adjustable overall image size
    for i, intensity_img in enumerate(region_intensity_images[:n]):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(intensity_img, cmap='gray')
        plt.title(f'{region_areas[i]}', fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_boundary_mask(shadow_mask, kernel_size=15):
    # Create an elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Dilate & Erode
    dilated = cv2.dilate(shadow_mask, kernel)
    eroded = cv2.erode(shadow_mask, kernel)
    # Compute boundary mask (Dilated - Eroded)
    boundary_mask = cv2.subtract(dilated, eroded)
    # Visualization
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.title("Original Shadow Mask")
    plt.imshow(shadow_mask, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title("Dilated")
    plt.imshow(dilated, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title("Eroded")
    plt.imshow(eroded, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title("Boundary Mask")
    plt.imshow(boundary_mask, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return boundary_mask


def compute_bhattacharyya_coefficient(hist1, hist2):
    # Assume hist1 and hist2 are already normalized
    return np.sum(np.sqrt(hist1 * hist2))


def compute_histogram(pixels_flat, channel, bins=32):
    # pixels_flat: (N, 3) numpy array
    channel_values = pixels_flat[:, channel]
    hist, _ = np.histogram(channel_values, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (np.sum(hist) + 1e-6)
    return hist


def dilate_mask(mask, kernel_size=3, iterations=1):
    """
    Dilates the mask to expand the shadow region.
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 127).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask


def global_illumination_correction(image, shadow_mask):
    # Calculate global illumination ratios
    global_ratios = []
    un_shaded_mask = np.uint8(shadow_mask == 0) # Black for non-shadow
    for i in range(3):  # For R, G, B channels
        shaded_light = shadow_mask * image[:, :, i]
        un_shaded_light = un_shaded_mask * image[:, :, i]
        shaded_illumination = np.mean(shaded_light[shadow_mask > 0]) + 1e-6
        un_shaded_illumination = np.mean(un_shaded_light[un_shaded_mask > 0]) + 1e-6
        global_ratio = un_shaded_illumination / shaded_illumination
        global_ratios.append(global_ratio)

    # Apply correction to shadow regions
    corrected_image = image.copy() # Create a copy
    shadow_area = np.expand_dims(shadow_mask, axis=2) # 2D to 3D
    global_ratios = np.array(global_ratios, dtype=np.float32) # Float type
    global_ratios_tiled = np.tile(global_ratios.reshape(1, 1, 3),
                                  (image.shape[0], image.shape[1], 1)) # Extend to same shape as image

    corrected_image = np.where(
        shadow_area > 0,
        np.clip(image * global_ratios_tiled, 0, 255).astype(np.uint8),
        image
    )
    return corrected_image

def segment_shadow_and_nonshadow(image, shadow_mask, pixel=600, compactness=20, sigma=1):
    """
    Performs SLIC superpixel segmentation separately for shadow and non-shadow regions.
    Parameters:
        image: np.ndarray Original RGB image
        shadow_mask: np.ndarray Binary mask (values 0 or 1)
        pixel: Helps control the number of superpixels
        compactness: Compactness parameter, higher value means more regular superpixels
        sigma: Smoothing parameter
    Returns:
        merged_labels: Merged superpixel label map (labels start from 1)
    """
    image_float = img_as_float(image)
    # Shadow and non-shadow masks
    shadow_area = (shadow_mask > 0)
    non_shadow_area = (shadow_mask == 0)

    # Segment shadow regions separately
    shadow_labels = np.zeros_like(shadow_mask, dtype=np.int32)  # Shadow superpixel labels
    if np.any(shadow_area):
        shadow_slic = slic(image_float, n_segments=int(np.sum(shadow_area) / pixel),
                            compactness=compactness, sigma=sigma, start_label=1, mask=shadow_area)
        shadow_labels[shadow_area] = shadow_slic[shadow_area]

    # Segment non-shadow regions separately
    non_shadow_labels = np.zeros_like(shadow_mask, dtype=np.int32)  # Non-shadow superpixel labels
    if np.any(non_shadow_area):
        non_shadow_slic = slic(image_float, n_segments=int(np.sum(non_shadow_area) / pixel),
                                compactness=compactness, sigma=sigma, start_label=1, mask=non_shadow_area)
        offset = shadow_labels.max()
        non_shadow_labels[non_shadow_area] = non_shadow_slic[non_shadow_area] + offset

    # Merge label maps
    merged_labels = shadow_labels + non_shadow_labels
    return merged_labels

def shadow_removal(image_path, shadow_mask_path, corrected_image_path, smooth_path):
    """
    Physical-based shadow removal for remote sensing images with global correction followed by superpixel optimization.
    """
    # Read input image and shadow mask
    image = imread(image_path)
    shadow_mask = imread_mask(shadow_mask_path)
    print(f"Processing image: {image_path}")

    # Step 1: Preliminary optimization using global illumination information
    image = global_illumination_correction(image, shadow_mask)
    print("Global illumination correction completed.")

    # Step 2: Superpixel segmentation
    segment_labels = segment_shadow_and_nonshadow(image, shadow_mask, compactness=10, sigma=1)
    print("Superpixel segmentation completed.")
    # Optional: Disable or comment out visualization if you don't want an image to pop up every time
    # plot_segmentation(image, segment_labels)

    # Calculate mask region properties
    regionsmask = measure.regionprops(segment_labels, intensity_image=shadow_mask)  # All mask superpixel regions
    region_is_shadows = np.array([r.mean_intensity > 0 for r in regionsmask])  # All mask shadow superpixels

    # Calculate global illumination ratios to limit exposure later
    global_ratios = []
    un_shaded_mask = np.uint8(shadow_mask == 0)
    for i in range(3):
        shaded_light = shadow_mask * image[:, :, i]
        un_shaded_light = un_shaded_mask * image[:, :, i]
        shaded_illumination = np.mean(shaded_light[shadow_mask > 0]) + 1e-6
        un_shaded_illumination = np.mean(un_shaded_light[un_shaded_mask > 0]) + 1e-6
        global_ratio = un_shaded_illumination / shaded_illumination
        global_ratios.append(global_ratio)

    # Calculate image region properties
    regionsimage = measure.regionprops(segment_labels, intensity_image=image)    # All image superpixel regions
    region_intensity_images = [r.intensity_image for r in regionsimage]  # RGB pixel values for each superpixel
    # region_areas = [r.area for r in regionsimage]    # Number of pixels for each superpixel - currently not used

    # Build adjacency matrix between superpixels
    graphs = graph.RAG(segment_labels)
    adjacency_matrix = nx.adjacency_matrix(graphs).todense()
    # adjacency_matrix_np = np.array(adjacency_matrix) # Currently not used

    # Initialize image; shadow regions are set to 0
    corrected_img = image * np.expand_dims(un_shaded_mask, axis=2)

    # Calculate centroids for all superpixels
    regionsxy = measure.regionprops(segment_labels)
    centroids = np.array([r.centroid for r in regionsxy])  # Shape (n, 2), n is number of superpixels, [y, x] coordinates

    # User-defined number of nearest neighbors
    numbers = 5  # Can be adjusted, e.g., 5 nearest non-shadow superpixels

    # Process each shadow superpixel
    region_shadow_indexs = np.where(region_is_shadows)[0] # Indices of shadow superpixels in regionsimage
    for region_shadow_index in tqdm(region_shadow_indexs):
        region_shadow_intensity_image = region_intensity_images[region_shadow_index]
        # region_shadow_area = region_areas[region_shadow_index] # Currently not used

        # Calculate mean pixel value for each channel
        shaded_illumination_bands = []
        for i in range(3):
            valid_pixels = region_shadow_intensity_image[:, :, i][region_shadow_intensity_image[:, :, i] > 0] # Valid pixels
            shaded_illumination_bands.append(np.mean(valid_pixels) if valid_pixels.size > 0 else 1e-6) # Mean pixel value per channel
        # Get centroid coordinates of the current shadow superpixel
        shadow_centroid = centroids[region_shadow_index]  # [y, x]
        # Get indices of all non-shadow superpixels
        non_shadow_indices = np.where(~region_is_shadows)[0]
        # Calculate Euclidean distance between current shadow superpixel and all non-shadow superpixels
        distances = np.sqrt(np.sum((centroids[non_shadow_indices] - shadow_centroid) ** 2, axis=1))
        # Find indices of the 'numbers' nearest non-shadow superpixels
        nearest_indices = non_shadow_indices[np.argsort(distances)[:numbers]]
        # Store indices of the nearest non-shadow superpixels in expanded_adj_indexs
        expanded_adj_indexs = set(nearest_indices.tolist())

        # Hyperparameters
        alpha = 0.6  # EMD weight
        beta = 0.3   # LBP Bhattacharyya distance weight
        gamma = 0.1  # a-channel mean difference weight

        # Convert shadow region to Lab and flatten
        # Avoid division by zero or operation on empty arrays
        if region_shadow_intensity_image.shape[0] == 0 or region_shadow_intensity_image.shape[1] == 0:
            continue # Skip invalid superpixel regions

        shadow_lab_pixels = rgb2lab(region_shadow_intensity_image / 255.0)
        # Filter valid pixels (e.g., non-zero pixels)
        valid_pixel_mask_shadow = (region_shadow_intensity_image[:, :, 0] > 0)
        shadow_lab_flat = shadow_lab_pixels[valid_pixel_mask_shadow].reshape(-1, 3)

        # Compute LBP for shadow region (using grayscale for simplicity)
        shadow_gray = np.mean(region_shadow_intensity_image, axis=2)  # Convert to grayscale
        # LBP requires images to be of uint8 or int type, or appropriate type conversion
        shadow_lbp = local_binary_pattern(shadow_gray.astype(np.uint8), P=8, R=1, method='uniform')
        shadow_lbp_flat = shadow_lbp[valid_pixel_mask_shadow].ravel()

        # Calculate a-channel mean for shadow region
        shadow_a_mean = np.mean(shadow_lab_flat[:, 1]) if shadow_lab_flat.shape[0] > 0 else 0

        # Initialize lists
        similarities = []
        un_shaded_illuminations = []

        # Evaluate candidate non-shadow regions
        for region_shadow_adj_index in expanded_adj_indexs:
            if not region_is_shadows[region_shadow_adj_index]:
                region_intensity_image = region_intensity_images[region_shadow_adj_index]

                # Avoid division by zero or operation on empty arrays
                if region_intensity_image.shape[0] == 0 or region_intensity_image.shape[1] == 0:
                    continue # Skip invalid neighboring regions

                # Convert non-shadow region to Lab and flatten
                region_lab_pixels = rgb2lab(region_intensity_image / 255.0)
                valid_pixel_mask_non_shadow = (region_intensity_image[:, :, 0] > 0)
                region_lab_flat = region_lab_pixels[valid_pixel_mask_non_shadow].reshape(-1, 3)

                # Compute LBP for non-shadow region
                region_gray = np.mean(region_intensity_image, axis=2)
                region_lbp = local_binary_pattern(region_gray.astype(np.uint8), P=8, R=1, method='uniform')
                region_lbp_flat = region_lbp[valid_pixel_mask_non_shadow].ravel()

                # Ensure arrays for EMD are not empty
                if shadow_lab_flat.shape[0] == 0 or region_lab_flat.shape[0] == 0:
                    emd = 0.0 # Or assign a large value to indicate dissimilarity
                else:
                    d_L = wasserstein_distance(shadow_lab_flat[:, 0], region_lab_flat[:, 0])
                    d_a = wasserstein_distance(shadow_lab_flat[:, 1], region_lab_flat[:, 1])
                    d_b = wasserstein_distance(shadow_lab_flat[:, 2], region_lab_flat[:, 2])
                    emd = (d_L + d_a + d_b) / 3.0

                # Ensure arrays for LBP histogram are not empty
                if shadow_lbp_flat.shape[0] == 0 or region_lbp_flat.shape[0] == 0:
                    lbp_bhat = 0.0 # Or assign a small value to indicate dissimilarity
                else:
                    hist1 = compute_histogram(shadow_lbp_flat.reshape(-1, 1), channel=0, bins=10) # LBP histogram (uniform pattern, 10 bins)
                    hist2 = compute_histogram(region_lbp_flat.reshape(-1, 1), channel=0, bins=10)
                    lbp_bhat = compute_bhattacharyya_coefficient(hist1, hist2)

                # a-channel mean difference
                region_a_mean = np.mean(region_lab_flat[:, 1]) if region_lab_flat.shape[0] > 0 else 0
                a_mean_diff = abs(shadow_a_mean - region_a_mean)

                # Normalize components (approximate scaling)
                emd_norm = emd / 100.0  # EMD typically has a large range, scale to [0, 1]
                lbp_bhat_norm = 1.0 - lbp_bhat  # Convert to distance, range [0, 1]
                a_mean_diff_norm = a_mean_diff / 128.0  # a-channel range [-128, 128], scale to [0, 1]

                # Combined distance score
                distance_combined = alpha * emd_norm + beta * lbp_bhat_norm + gamma * a_mean_diff_norm
                similarity = 1.0 / (distance_combined + 1e-6)  # Smaller distance means higher similarity
                similarities.append(similarity)

                # Illumination in non-shadow region
                illumination_bands = []
                for i in range(3):
                    valid_pixels = region_intensity_image[:, :, i][region_intensity_image[:, :, i] > 0]
                    illumination_bands.append(np.mean(valid_pixels) if valid_pixels.size > 0 else 1e-6)
                un_shaded_illuminations.append(illumination_bands)

        # Normalize similarities to weights
        similarities = np.array(similarities)
        if len(similarities) > 0:
            weights = similarities / np.maximum(np.sum(similarities), 1e-6)
            if np.sum(weights) == 0:
                # If all weights are 0, fall back to global ratios
                ratio_bands = np.array(global_ratios)
            else:
                un_shaded_illumination_bands = np.average(np.array(un_shaded_illuminations), axis=0, weights=weights)
                shaded_illumination_bands = np.array(shaded_illumination_bands)
                ratio_bands = un_shaded_illumination_bands / (shaded_illumination_bands + 1e-6)
        else:
            # If no suitable neighboring regions are found, use global ratios
            ratio_bands = np.array(global_ratios)

        # Limit ratios to prevent over-amplification
        for i in range(3):
            if ratio_bands[i] > global_ratios[i]:
                ratio_bands[i] = global_ratios[i]
            if ratio_bands[i] < 1:
                ratio_bands[i] = 1

        # Apply correction to shadow region
        # Ensure that segment_labels == (region_shadow_index + 1) results in a boolean mask
        current_region_mask = (segment_labels == (region_shadow_index + 1))
        # Check if the current region is empty, if so, skip
        if not np.any(current_region_mask):
            continue

        image_region = image * np.expand_dims(current_region_mask, axis=2)
        ratio_bands_tiled = np.tile(ratio_bands.reshape(1, 1, -1), (image.shape[0], image.shape[1], 1))
        image_region_correct = np.clip(image_region * ratio_bands_tiled, 0, 255).astype(np.uint8)
        corrected_img = np.where(np.expand_dims(current_region_mask, axis=2), image_region_correct, corrected_img)

    imwrite(corrected_img, corrected_image_path)
    print("Shadow removal completed.")

    # Step 3: Boundary optimization
    # The boundary_smooth function will read corrected_image_path internally and process it
    final_image = boundary_smooth(
        corrected_image_path,
        shadow_mask_path,
    )
    print("Shadow boundary smoothing completed.")

    return final_image


def main(image_path, shadow_mask_path, output_dir):
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    os.makedirs(output_dir, exist_ok=True)

    corrected_image_path = get_deshadowed_path(output_dir, base_name, ext)
    # The smooth_path variable still needs to be passed to shadow_removal,
    # but the file it points to will no longer be saved.
    smooth_path = get_smooth_path(output_dir, base_name, ext)

    final_image = shadow_removal(image_path, shadow_mask_path,
                                 corrected_image_path, smooth_path)

    # Save the final image here, named with _final suffix
    final_output_path = os.path.join(output_dir, f"{base_name}_final{ext}")
    cv2.imwrite(final_output_path, cv2.cvtColor(final_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Final deshadowed image saved to: {final_output_path}")

    return final_image, ext


def batch_process_shadow_images(input_dir, mask_dir, output_dir):
    """
    Batch processes all .tif or .png images in the 'shadow' folder and their corresponding masks.

    Parameters:
        input_dir (str): Directory for shadow images
        mask_dir (str): Directory for masks
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_path in glob.glob(os.path.join(input_dir, "*")):
        if not (image_path.lower().endswith(".tif") or image_path.lower().endswith(".png")):
            continue  # Skip non-image files

        base_name, ext = os.path.splitext(os.path.basename(image_path))
        print(f"Found image: {base_name}{ext}") # Print the name of the image being processed
        # Adjust to use the same extension as the original image for mask, or modify as needed
        shadow_mask_path = os.path.join(mask_dir, f"{base_name}{ext}")

        if not os.path.exists(shadow_mask_path):
            print(f"⚠ Warning: Mask file {shadow_mask_path} not found, skipping {base_name}.")
            continue

        try:
            # The main function is now responsible for saving the _final image
            final_image, _ = main(image_path, shadow_mask_path, output_dir)
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")


if __name__ == "__main__":
    input_dir = r"D:\BaiduNetdiskDownload\AISD\Test51\shadow"
    mask_dir = r"D:\BaiduNetdiskDownload\AISD\Test51\mask"
    output_dir = r"D:\BaiduNetdiskDownload\AISD\Test51\removeshadow"
    batch_process_shadow_images(input_dir, mask_dir, output_dir)