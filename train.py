'''没有定下随机种子，最近一次指标达到了
Mean Accuracy:  97.76%
Mean Precision: 96.01%
Mean Recall:    93.59%
Mean F1 Score:  94.73%
Mean BER:       3.76%
Mean IoU:       90.04%
'''
import os
# set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from Dataset.dataset import ShadowDataset
from utils.DBSCF_DCENet import ShadowDetectionNetwork
from collections import deque

# --------------------- 参数设置 ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--cuda', default='true', action='store_true', help='use GPU if available')
parser.add_argument('--n_epochs', type=int, default=30, help='total number of training epochs')
parser.add_argument('--batchSize', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=15, help='linear decay start epoch')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='only save best model in last N epochs')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use')
parser.add_argument('--train_image_dir', type=str, default=r"D:\BaiduNetdiskDownload\AISD\Test51\shadow", help='path to training images')
parser.add_argument('--train_mask_dir', type=str, default=r"D:\BaiduNetdiskDownload\AISD\Test51\mask", help='path to training masks')
parser.add_argument('--val_image_dir', type=str, default=r"D:\BaiduNetdiskDownload\AISD\Val51\shadow", help='path to validation images')
parser.add_argument('--val_mask_dir', type=str, default=r"D:\BaiduNetdiskDownload\AISD\Val51\mask", help='path to validation masks')
parser.add_argument('--save_dir', type=str, default="./source_checkpoint", help='directory to save models')
opt = parser.parse_args()


if torch.cuda.is_available():
    opt.cuda = True
device = torch.device("cuda:0" if opt.cuda else "cpu")
os.makedirs(opt.save_dir, exist_ok=True)


# --------------------- 学习率调度器 ---------------------
class LinearLRScheduler:
    def __init__(self, optimizer, initial_lr, decay_start_epoch, total_epochs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_start_epoch = decay_start_epoch
        self.total_epochs = total_epochs

    def step(self, current_epoch):
        if current_epoch < self.decay_start_epoch:
            return
        decay_ratio = (current_epoch - self.decay_start_epoch) / (self.total_epochs - self.decay_start_epoch)
        lr = self.initial_lr * (1 - decay_ratio)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(f">> Adjusted learning rate to: {lr:.6f}")


# --------------------- 训练主函数 ---------------------

def train(model, train_loader, val_loader, device, opt):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = LinearLRScheduler(optimizer, opt.lr, opt.decay_epoch, opt.n_epochs)
    saved_models = deque(maxlen=opt.snapshot_epochs)

    log_path = os.path.join(opt.save_dir, "training_log.txt")

    for epoch in range(opt.epoch, opt.n_epochs):
        if epoch == opt.epoch:  # 第一次训练，写入表头
            with open(log_path, 'w') as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")

        model.train()
        running_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{opt.n_epochs}] - Training")
        for images, masks in tqdm(train_loader, desc="Training", leave=False):
            images, masks = images.to(device), masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}], Train Loss: {avg_train_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        print(f"Epoch [{epoch+1}] - Validation")
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating", leave=False):
                images, masks = images.to(device), masks.to(device)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                masks = (masks > 0.5).float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}], Val Loss: {avg_val_loss:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\n")

        if epoch + 1 > 0:
            model_path = os.path.join(opt.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            saved_models.append(model_path)
            print(f">> Saved checkpoint at Epoch {epoch+1}")

        scheduler.step(epoch + 1)



# --------------------- prepare data and training ---------------------
if __name__ == "__main__":
    print(opt)
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_dataset = ShadowDataset(
        image_dir=opt.train_image_dir,
        mask_dir=opt.train_mask_dir,
        transform=train_transform,
        mask_transform=mask_transform,
    )

    val_dataset = ShadowDataset(
        image_dir=opt.val_image_dir,
        mask_dir=opt.val_mask_dir,
        transform=train_transform,
        mask_transform=mask_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    model = ShadowDetectionNetwork().to(device)
    train(model, train_loader, val_loader, device, opt)
