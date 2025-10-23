import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#from utils.model import ShadowDetectionNet
from utils.DBSCF_DCENet import ShadowDetectionNetwork
from Dataset.dataset import ShadowDataset

# ---------- Configuration Parameters ----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # Option to force CPU usage
checkpoint_path = r"D:\Desktop\SARU-Net\bestcheckpoint\best_AISD_ckp.pth"
test_img_dir = r"D:\BaiduNetdiskDownload\AISD\Test51\shadow"
test_mask_dir = r"D:\BaiduNetdiskDownload\AISD\Test51\mask"
output_dir = r"D:\BaiduNetdiskDownload\AISD\Test51\sdresult"
os.makedirs(output_dir, exist_ok=True)
canvas_dir = os.path.join(output_dir, "canvas")
os.makedirs(canvas_dir, exist_ok=True)
batch_size = 1

# ---------- Image Transformations ----------
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
mask_transform = transforms.Compose([
    transforms.Resize((512,512), interpolation = transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

# ---------- Metric Calculation Function ----------
def Evaluator(pred_mask, gt_mask):
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()

    TP = np.sum((pred == 255) & (gt == 255))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 255) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 255))

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0
    ber = 1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP)) if (TP + FN > 0 and TN + FP > 0) else 1
    iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0

    return accuracy, precision, recall, f1, ber, iou

# ---------- Canvas Function ----------
def save_canvas(shadow_img, gt_mask, pred_mask, save_path):
    shadow_img = shadow_img.convert("RGB")
    gt_img = Image.fromarray(gt_mask).convert("RGB")
    pred_img = Image.fromarray(pred_mask).convert("RGB")

    w, h = shadow_img.size
    canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
    canvas.paste(shadow_img, (0, 0))
    canvas.paste(gt_img, (w, 0))
    canvas.paste(pred_img, (w * 2, 0))

    # Add titles
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text((w//2 - 40, 5), "Shadow", fill="black", font=font)
    draw.text((w + w//2 - 20, 5), "GT", fill="black", font=font)
    draw.text((2 * w + w//2 - 30, 5), "Mask", fill="black", font=font)

    canvas.save(save_path)

# ---------- Load Data ----------
test_dataset = ShadowDataset(
    image_dir=test_img_dir,
    mask_dir=test_mask_dir,
    transform=transform,
    mask_transform=mask_transform,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------- Load Model ----------
model = ShadowDetectionNetwork().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ---------- Inference ----------
print("Starting inference on the test set...")
results_file = os.path.join(output_dir, "results.txt")
all_acc, all_pre, all_rec, all_f1, all_ber, all_iou = [], [], [], [], [], []

with open(results_file, "w") as f:
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image = image.to(device)
            image_name = test_dataset.image_files[i]
            file_name, _ = os.path.splitext(image_name)

            # Restore original image for canvas visualization
            shadow_img = Image.open(os.path.join(test_img_dir, image_name)).convert("RGB")
            gt = (mask.squeeze().numpy() * 255).astype(np.uint8)

            pred_mask = model(image)
            pred_mask = pred_mask.squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

            # Save predicted mask image
            mask_save_path = os.path.join(output_dir, file_name + '_mask.png')
            Image.fromarray(pred_mask).save(mask_save_path)

            # Save comparison canvas
            canvas_save_path = os.path.join(canvas_dir, file_name + '_canvas.png')
            save_canvas(shadow_img, gt, pred_mask, canvas_save_path)

            # Calculate metrics
            acc, pre, rec, f1, ber, iou = Evaluator(pred_mask, gt)
            all_acc.append(acc)
            all_pre.append(pre)
            all_rec.append(rec)
            all_f1.append(f1)
            all_ber.append(ber)
            all_iou.append(iou)

            # Write results to file
            f.write(f"{image_name}: Accuracy={acc * 100:.2f}% Precision={pre * 100:.2f}% Recall={rec * 100:.2f}% "
                    f"F1={f1 * 100:.2f}% BER={ber * 100:.2f}% IoU={iou * 100:.2f}%\n")
            print(f"✅ {image_name} processed | Accuracy={acc * 100:.2f}%")

    # ---------- Output Average Metrics ----------
    f.write("\n====== Average Metrics ======\n")
    f.write(f"Mean Accuracy:  {np.mean(all_acc) * 100:.2f}%\n")
    f.write(f"Mean Precision: {np.mean(all_pre) * 100:.2f}%\n")
    f.write(f"Mean Recall:    {np.mean(all_rec) * 100:.2f}%\n")
    f.write(f"Mean F1 Score:  {np.mean(all_f1) * 100:.2f}%\n")
    f.write(f"Mean BER:       {np.mean(all_ber) * 100:.2f}%\n")
    f.write(f"Mean IoU:       {np.mean(all_iou) * 100:.2f}%\n")

print("\n🎯 Inference finished, metrics recorded in results.txt")