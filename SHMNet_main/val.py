import argparse
import torch
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from module.CustomDataset import DataLoad
from sklearn.metrics import f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.models.swin_transformer import swin_s
from torchvision.models.convnext import convnext_small
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.resnet import resnet50
from torchvision.models.efficientnet import efficientnet_v2_l
from torchvision.models.vision_transformer import vit_b_16
from models.SHMNet import SHMNet1_1, SHMNet1_2, SHMNet1_3, SHMNet2_1, SHMNet2_2, SHMNet2_3
from tqdm import tqdm

cfgs = {
    'resnet50': resnet50,
    'mobilenet_l': mobilenet_v3_large,
    'convnext_s': convnext_small,
    'ef2l': efficientnet_v2_l,
    'swin_s': swin_s,
    'vit_b_16': vit_b_16,

    'SHMNet1_1': SHMNet1_1,
    'SHMNet1_2': SHMNet1_2,
    'SHMNet1_3': SHMNet1_3,
    'SHMNet2_1': SHMNet2_1,
    'SHMNet2_2': SHMNet2_2,
    'SHMNet2_3': SHMNet2_3,

       }

def calssicmodel_using_name(model_name):
    return cfgs[model_name]

parser = argparse.ArgumentParser(description='Model Validation Script')
parser.add_argument('--model', type=str, default='resnet50', help='select a model for validation')
parser.add_argument('--weights', type=str, default='results/exp2/resnet50_train1_results/best_model.pth')
parser.add_argument('--num_classes', type=int, default=6, help='the number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size for validation')
parser.add_argument('--data_path', type=str, default=r"D:\artist6", help='path to dataset directory')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--results_dir', type=str, default='results/test', help='base results directory')
parser.add_argument('--seed', action='store_true', help='fix random seed')


def main(args):
    if args.seed:
        import random
        def seed_torch(seed=42):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            print('Random seed fixed')

        seed_torch()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Validation Configuration:")
    for k, v in vars(args).items():
        print(f"{k:>15}: {v}")

    # Load model---------------------------------------------------------------------
    model = calssicmodel_using_name(opt.model)(num_classes=opt.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    print(f"Model loaded on {device}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Data transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    val_dataset = DataLoad(os.path.join(args.data_path, 'test'), transform=val_transform)
    if args.num_classes != val_dataset.num_class:
        raise ValueError(f"Dataset has {val_dataset.num_class} classes, but model expects {args.num_classes}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    # Create specific results subdirectory
    val_num = 1
    results_subdir = f"{args.model}_val{val_num}_results"
    results_path = os.path.join(args.results_dir, results_subdir)

    # Handle existing directories like original script
    while os.path.exists(results_path):
        val_num += 1
        results_subdir = f"{args.model}_val{val_num}_results"
        results_path = os.path.join(args.results_dir, results_subdir)

    os.makedirs(results_path, exist_ok=True)
    print(f"All results will be saved to {results_path}")

    # Save arguments to file
    args_path = os.path.join(results_path, 'args.txt')
    with open(args_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    # Validation with progress bar
    model.eval()
    all_labels = []
    all_preds = []

    print("\nStarting validation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Save metrics to CSV
    metrics = {
        'Accuracy': [accuracy],
        'F1_Score': [f1],
        'Recall': [recall]
    }

    csv_path = os.path.join(results_path, 'validation_metrics.csv')
    pd.DataFrame(metrics).to_csv(csv_path, index=False)
    print(f"\nValidation metrics saved to {csv_path}")

    # Confusion matrix
    cm_path = os.path.join(results_path, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    print("\nValidation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall:   {recall:.4f}")


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)