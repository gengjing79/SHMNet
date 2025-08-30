import argparse
import torch
import math
from tqdm import tqdm
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
import models
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="SHMNet2_1", help=' select a model for training')
parser.add_argument('--weights1', type=str, default="results/exp2/swin_s_train1_results/best_model.pth")
parser.add_argument('--weights2', type=str, default="results/exp2/mobilenet_l_train1_results/best_model.pth")
parser.add_argument('--weights3', type=str, default="results/exp2/convnext_s_train1_results/best_model.pth")
parser.add_argument('--num_classes', type=int, default=6, help='the number of classes')
parser.add_argument('--epochs', type=int, default=50, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.00005, help='end learning rate')
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

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    # 创建模型
    model = models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device)
    model.branch1.load_state_dict(torch.load(opt.weights1, map_location=device))
    model.branch2.load_state_dict(torch.load(opt.weights2, map_location=device))
    #model.branch3.load_state_dict(torch.load(opt.weights3, map_location=device))

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
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    train_dataset = DataLoad(os.path.join(args.data_path, 'train'), transform=data_transform["train"])
    val_dataset = DataLoad(os.path.join(args.data_path, 'test'), transform=data_transform["val"])

    if args.num_classes != train_dataset.num_class:#检查类别数和数据集类别是否相同
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
    single_ratios = []
    multi_ratios = []

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

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        single_count = 0
        multi_count = 0

        with tqdm(train_loader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                final_logits, selection_probs, auxiliary_losses = model(images, labels)

                loss = (criterion(final_logits, labels)+0.01 * auxiliary_losses["balance_loss"] +
            0.1 * auxiliary_losses["incentive_loss"])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                max_prob_index = torch.argmax(selection_probs, dim=1)
                single_count += torch.sum(max_prob_index == 0).item()
                multi_count += torch.sum(max_prob_index == 1).item()

                accuracy = (final_logits.argmax(dim=1) == labels).float().mean()

                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        total_samples = single_count + multi_count
        single_ratio = single_count / total_samples
        multi_ratio = multi_count / total_samples
        single_ratios.append(single_ratio)
        multi_ratios.append(multi_ratio)

        model.eval()
        val_loss = 0
        val_accuracy = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                final_logits = model(images)
                val_loss += criterion(final_logits, labels).item()

                predicted = final_logits.argmax(dim=1)
                val_accuracy += (predicted == labels).float().mean().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')

        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            best_model_path = os.path.join(results_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        total_samples = single_count + multi_count
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}")
        print(
            f"Epoch {epoch + 1}, single: {single_count / total_samples:.2f}, multi: {multi_count / total_samples:.2f}")
        data = pd.DataFrame({
            'train_loss': [train_losses[-1]],
            'val_loss': [val_losses[-1]],
            'val_acc': [val_accs[-1]],
            'val_f1': [val_f1s[-1]],
            'val_recall': [val_recalls[-1]],
            'single_ratio': [single_ratios[-1]],
            'multi_ratio': [multi_ratios[-1]],
        })

        data.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
if __name__ == "__main__":
    main(opt)
