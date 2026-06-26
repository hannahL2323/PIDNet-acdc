import argparse
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

# Add PIDNet root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models


def parse_args():
    parser = argparse.ArgumentParser(description='ACDC Evaluation')
    parser.add_argument('--data_root', type=str, default='/home/deng/datasets/acdc_mmseg')
    parser.add_argument('--model_path', type=str, 
                        default='/home/deng/PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt')
    parser.add_argument('--num_classes', type=int, default=19)
    return parser.parse_args()


def get_confusion_matrix(gt_label, pred_label, num_classes):
    mask = (gt_label >= 0) & (gt_label < num_classes)
    hist = np.bincount(
        num_classes * gt_label[mask].astype(int) + pred_label[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def main():
    args = parse_args()
    
    # Build model
    model = models.pidnet.PIDNet(m=2, n=3, num_classes=args.num_classes, 
                                  planes=32, ppm_planes=96, head_planes=128, augment=True)
    
    # Load pretrained weights
    pretrained = torch.load(args.model_path, map_location='cpu')
    if 'state_dict' in pretrained:
        pretrained = pretrained['state_dict']
    model_dict = model.state_dict()
    pretrained = {k[6:]: v for k, v in pretrained.items() if k[6:] in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    
    model = model.cuda()
    model.eval()
    
    # Preprocessing params
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Collect images grouped by weather
    data_root = args.data_root
    img_dir = os.path.join(data_root, 'leftImg8bit', 'val')
    label_dir = os.path.join(data_root, 'gtFine', 'val')
    
    weather_images = defaultdict(list)
    for f in sorted(os.listdir(img_dir)):
        if f.endswith('.png'):
            weather = f.split('_')[0]
            weather_images[weather].append(f)
    
    # Store per-weather and overall confusion matrix
    weather_confusion = {}
    overall_confusion = np.zeros((args.num_classes, args.num_classes))
    
    # Evaluate per weather condition
    for weather, files in weather_images.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {weather} ({len(files)} images)")
        print(f"{'='*50}")
        
        confusion_matrix = np.zeros((args.num_classes, args.num_classes))
        
        with torch.no_grad():
            for fname in tqdm(files, desc=weather):
                img_path = os.path.join(img_dir, fname)
                label_name = fname.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                label_path = os.path.join(label_dir, label_name)
                
                # Load image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                
                # Preprocess
                img = img / 255.0
                img = (img - mean) / std
                img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda()
                
                # Inference
                output = model(img)
                if isinstance(output, list):
                    output = output[1]
                
                output = nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                pred = output.argmax(1).squeeze().cpu().numpy()
                
                # Load label (ACDC uses trainIds already)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                
                # Update confusion matrices
                hist = get_confusion_matrix(label, pred, args.num_classes)
                confusion_matrix += hist
                overall_confusion += hist
        
        weather_confusion[weather] = confusion_matrix
        
        # Compute metrics
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-10)
        miou = np.nanmean(iou)
        
        print(f"mIoU: {miou*100:.2f}%")
        
        class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                       'traffic light', 'traffic sign', 'vegetation', 'terrain',
                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                       'motorcycle', 'bicycle']
        print("\nPer-Class IoU:")
        for i, name in enumerate(class_names):
            print(f"  {name:15s}: {iou[i]*100:.2f}%")
    
    # Overall results
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS (All Weather)")
    print(f"{'='*60}")
    
    overall_iou = np.diag(overall_confusion) / (overall_confusion.sum(axis=1) + overall_confusion.sum(axis=0) - np.diag(overall_confusion) + 1e-10)
    overall_miou = np.nanmean(overall_iou)
    total_images = sum(len(files) for files in weather_images.values())
    
    print(f"\nOverall mIoU: {overall_miou*100:.2f}%")
    print(f"Total Images: {total_images}")
    print(f"\nOverall Per-Class IoU:")
    for i, name in enumerate(class_names):
        print(f"  {name:15s}: {overall_iou[i]*100:.2f}%")
    
    # Per-weather summary
    print(f"\n{'='*60}")
    print(f"SUMMARY BY WEATHER")
    print(f"{'='*60}")
    for weather in weather_images:
        conf = weather_confusion[weather]
        iou = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf) + 1e-10)
        miou = np.nanmean(iou)
        print(f"  {weather:8s}: mIoU = {miou*100:.2f}%")


if __name__ == '__main__':
    main()