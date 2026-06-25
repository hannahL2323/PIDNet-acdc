import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import cv2

# Add PIDNet root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import models


class ACDCDataset(Dataset):
    classes = (
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle"
    )
    
    IGNORE_INDEX = 255

    def __init__(self, data_dir: str, split: str = "train", 
                 crop_size: tuple = None, is_training: bool = False):
        self.crop_size = crop_size
        self.is_training = is_training
        self.samples = []
        
        img_dir = os.path.join(data_dir, "leftImg8bit", split)
        label_dir = os.path.join(data_dir, "gtFine", split)
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Cannot find images directory: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Cannot find labels directory: {label_dir}")
        
        print(f"Loading {split} set from {img_dir}")
        
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(('.png', '.jpg')):
                continue
            img_path = os.path.join(img_dir, fname)
            label_name = fname.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
            label_path = os.path.join(label_dir, label_name)
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))
        
        print(f"ACDC {split} set: found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path))
        label = label.astype(np.int64)
        label[(label < 0) | (label >= 19)] = self.IGNORE_INDEX
        
        if self.is_training and self.crop_size is not None:
            scale = random.uniform(0.5, 2.0)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()
            
            if random.random() < 0.3:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-0.1, 0.1)
                image = np.clip(alpha * image + beta * 255, 0, 255).astype(np.uint8)
            
            crop_h, crop_w = self.crop_size
            h, w = image.shape[:2]
            
            if h < crop_h or w < crop_w:
                pad_h = max(0, crop_h - h)
                pad_w = max(0, crop_w - w)
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 
                             mode='constant', constant_values=0)
                label = np.pad(label, ((0, pad_h), (0, pad_w)), 
                             mode='constant', constant_values=self.IGNORE_INDEX)
                h, w = image.shape[:2]
            
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            image = image[top:top+crop_h, left:left+crop_w]
            label = label[top:top+crop_h, left:left+crop_w]
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])(image)
        label = torch.from_numpy(label).long()
        
        return {"data": image, "label": label}


def evaluate(model, dataloader, num_classes=19, device="cuda"):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.inference_mode():
        for feed_dict in tqdm(dataloader, desc="Evaluating", leave=False):
            images = feed_dict["data"].to(device)
            masks = feed_dict["label"].to(device)
            
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[1]
            
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                       mode='bilinear', align_corners=True)
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            
            for pred, target in zip(preds, targets):
                mask = (target != 255)
                pred = pred[mask]
                target = target[mask]
                confusion_matrix += np.bincount(
                    target * num_classes + pred, 
                    minlength=num_classes**2
                ).reshape(num_classes, num_classes)
    
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        intersection = confusion_matrix[cls, cls]
        union = (confusion_matrix[cls, :].sum() + 
                confusion_matrix[:, cls].sum() - 
                confusion_matrix[cls, cls])
        if union > 0:
            iou_per_class[cls] = intersection / union
    
    miou = np.nanmean(iou_per_class) if np.any(iou_per_class > 0) else 0.0
    return miou * 100, iou_per_class * 100


def train(model, train_loader, val_loader, optimizer, criterion, 
          scheduler, total_iters, device, save_dir, 
          log_interval=100, val_interval=2000, checkpoint_interval=2000):
    model.train()
    global_step = 0
    total_loss = 0
    best_miou = 0.0
    
    print(f"\nStarting training for {total_iters} iterations...")
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Log interval: {log_interval} iters")
    print(f"Val interval: {val_interval} iters")
    print(f"Checkpoint interval: {checkpoint_interval} iters")
    start_time = time.time()
    
    train_iter = iter(train_loader)
    
    while global_step < total_iters:
        try:
            feed_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            feed_dict = next(train_iter)
        
        images = feed_dict["data"].to(device)
        masks = feed_dict["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if isinstance(outputs, list):
            output_main = outputs[1]
            if output_main.shape[-2:] != masks.shape[-2:]:
                output_main = F.interpolate(output_main, size=masks.shape[-2:],
                                           mode='bilinear', align_corners=True)
            loss = criterion(output_main, masks)
        else:
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                       mode='bilinear', align_corners=True)
            loss = criterion(outputs, masks)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        global_step += 1
        
        if global_step % log_interval == 0:
            avg_loss = total_loss / global_step
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"Iter {global_step}/{total_iters} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s")
        
        if global_step % val_interval == 0 or global_step == total_iters:
            print(f"\n=== Evaluating at iteration {global_step} ===")
            miou, iou_per_class = evaluate(model, val_loader, device=device)
            print(f"Val mIoU: {miou:.2f}%")
            
            print("\nPer-class IoU:")
            classes = train_loader.dataset.classes
            for i, cls_name in enumerate(classes):
                if iou_per_class[i] > 0:
                    print(f"  {cls_name:15s}: {iou_per_class[i]:.2f}%")
            
            if miou > best_miou:
                best_miou = miou
                checkpoint = {
                    'iteration': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': miou,
                    'iou_per_class': iou_per_class,
                }
                torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
                print(f"Best model saved! mIoU: {miou:.2f}%")
            
            model.train()
        
        if global_step % checkpoint_interval == 0:
            checkpoint = {
                'iteration': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': miou if 'miou' in locals() else 0.0,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_iter_{global_step}.pth"))
            print(f"Checkpoint saved at iter {global_step}")
    
    return best_miou


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PIDNet-S on ACDC")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to ACDC dataset (e.g., /home/deng/datasets/acdc_mmseg)")
    parser.add_argument("--pretrained", type=str, 
                       default="/home/deng/PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt",
                       help="Path to pretrained PIDNet-S Cityscapes weights")
    parser.add_argument("--gpu", type=str, default="0",
                       help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--iters", type=int, default=40000,
                       help="Total training iterations")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Initial learning rate")
    parser.add_argument("--crop_height", type=int, default=512,
                       help="Crop height for training")
    parser.add_argument("--crop_width", type=int, default=1024,
                       help="Crop width for training")
    parser.add_argument("--save_dir", type=str, 
                       default="/home/deng/PIDNet/checkpoints/acdc_finetune",
                       help="Directory to save checkpoints")
    parser.add_argument("--warmup_iters", type=int, default=1500,
                       help="Warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Logging interval (iters)")
    parser.add_argument("--val_interval", type=int, default=2000,
                       help="Validation interval (iters)")
    parser.add_argument("--checkpoint_interval", type=int, default=2000,
                       help="Checkpoint saving interval (iters)")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Build PIDNet-S model
    print(f"\nBuilding PIDNet-S model...")
    model = models.pidnet.PIDNet(m=2, n=3, num_classes=19, 
                                  planes=32, ppm_planes=96, head_planes=128, 
                                  augment=True)
    
    # Load pretrained Cityscapes weights
    print(f"Loading pretrained weights from {args.pretrained}")
    pretrained = torch.load(args.pretrained, map_location='cpu')
    if 'state_dict' in pretrained:
        pretrained = pretrained['state_dict']
    model_dict = model.state_dict()
    # Remove 'module.' prefix (6 chars) from pretrained keys
    pretrained = {k[6:]: v for k, v in pretrained.items() if k[6:] in model_dict}
    msg = f'Loaded {len(pretrained)} parameters from pretrained model'
    print(msg)
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    
    model = model.to(device)
    print("Model loaded successfully")
    
    # Load data
    print("\nLoading ACDC dataset...")
    train_crop_size = (args.crop_height, args.crop_width)
    
    train_dataset = ACDCDataset(args.data_path, split="train", 
                               crop_size=train_crop_size, is_training=True)
    # Validation: no crop, use full resolution
    val_dataset = ACDCDataset(args.data_path, split="val", 
                             crop_size=None, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=ACDCDataset.IGNORE_INDEX)
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.weight_decay)
    
    # Poly learning rate schedule
    power = 0.9
    def poly_lr_lambda(current_iter):
        if current_iter < args.warmup_iters:
            return float(current_iter) / float(max(1, args.warmup_iters))
        else:
            progress = float(current_iter - args.warmup_iters) / float(max(1, args.iters - args.warmup_iters))
            return (1.0 - progress) ** power
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: PIDNet-S")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Data path: {args.data_path}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  GPU: {args.gpu}")
    print(f"  LR schedule: Poly (power={power}) + Warmup ({args.warmup_iters} iters)")
    print(f"  Initial LR: {args.lr}")
    print(f"  Total iters: {args.iters}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train crop size: {train_crop_size}")
    print(f"  Val: full resolution (no crop)")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Log interval: {args.log_interval}")
    print(f"  Val interval: {args.val_interval}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"{'='*60}\n")
    
    # Train
    best_miou = train(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler, args.iters, device, args.save_dir,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval
    )
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best mIoU: {best_miou:.2f}%")
    print(f"Checkpoints saved in: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()