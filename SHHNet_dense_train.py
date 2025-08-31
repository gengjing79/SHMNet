import argparse
import torch
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pandas as pd
import random
import numpy as np
import torchvision.transforms as transforms
from module.CustomDataset import DataLoad
from sklearn.metrics import f1_score, recall_score
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import models
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="SHHNet1_3", help=' select a model for training')
parser.add_argument('--weights1', type=str, default="results/exp2/swin_s_train1_results/best_model.pth")
parser.add_argument('--weights2', type=str, default="results/exp2/mobilenet_l_train1_results/best_model.pth")
parser.add_argument('--weights3', type=str, default="results/exp2/convnext_s_train1_results/best_model.pth")
parser.add_argument('--num_classes', type=int, default=6, help='the number of classes')
parser.add_argument('--epochs', type=int, default=50, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.001, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0005, help='end learning rate')
parser.add_argument('--data_path', type=str, default=r"D:\artist6", help='path to dataset')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--results_dir', type=str, default='results/test', help='results dir')
parser.add_argument('--seed', default=True, action='store_true', help='fix the initialization of parameters')
opt = parser.parse_args()
if opt.seed:
    def seed_torch(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('random seed has been fixed')
    seed_torch()
#-----------------------------------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    model = models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device)
    model.branch1.load_state_dict(torch.load(opt.weights1, map_location=device))
    model.branch2.load_state_dict(torch.load(opt.weights2, map_location=device))
    # model.branch3.load_state_dict(torch.load(opt.weights3, map_location=device))

    for param in model.branch1.parameters():
        param.requires_grad = False

    for param in model.branch2.parameters():
        param.requires_grad = False

    #for param in model.branch3.parameters():
    #param.requires_grad = False

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    train_dataset = DataLoad(os.path.join(args.data_path, 'train'), transform=data_transform["train"])
    val_dataset = DataLoad(os.path.join(args.data_path, 'test'), transform=data_transform["test"])

    if args.num_classes != train_dataset.num_class:
        raise ValueError("dataset have {} classes, but input {}".format(train_dataset.num_class, args.num_classes))
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-3)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    criterion = nn.CrossEntropyLoss()

    patience = 5
    best_val_loss = float('inf')
    counter = 0

    val_accs = []
    train_losses = []
    val_losses = []
    val_f1s = []
    val_recalls = []
    all_labels = []
    all_preds = []

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        print('could not find your results dir')
    train_num = 1
    results_path_name = f'{args.model}_train{train_num}_results'
    results_path = os.path.join(results_dir, results_path_name)
    while os.path.exists(results_path):
        train_num += 1
        results_path_name = f'{args.model}_train{train_num}_results'
        results_path = os.path.join(results_dir, results_path_name)
    os.makedirs(results_path, exist_ok=True)
    csv_path = os.path.join(results_path, 'results.csv')
    args_path = os.path.join(results_path, 'args.txt')
    with open(args_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

#---------------------------------------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        with tqdm(train_loader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.to(device)
                optimizer.zero_grad()
                output = model(images)

                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                output = model(images)
                predicted = output.argmax(dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loss += criterion(output, labels.to(device)).item()  # 累积验证损失和准确率。
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy /= len(val_loader)
        val_accs.append(val_accuracy)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            if not os.path.exists(results_path):
                os.makedirs(results_path, exist_ok=True)
            best_model_path = os.path.join(results_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        data = pd.DataFrame({
            'train_loss': [train_losses[-1]],
            'val_loss': [val_losses[-1]],
            'val_acc': [val_accs[-1]],
            'val_f1': [val_f1s[-1]],
            'val_recall': [val_recalls[-1]]
        })
        data.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
if __name__ == "__main__":
    main(opt)
