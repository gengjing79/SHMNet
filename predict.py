from PIL import Image
import torch
import os
import torchvision.transforms as transforms
import csv
import argparse
from torchvision.models.swin_transformer import swin_s
from torchvision.models.convnext import convnext_small
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.resnet import resnet50
from torchvision.models.efficientnet import efficientnet_v2_s
from models.SHHNet import SHHNet1_1, SHHNet1_2, SHHNet1_3, SHHNet2_1, SHHNet2_2, SHHNet2_3

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=SHHNet2_1, help='select a model for validation')
parser.add_argument('--path', type=str, default="data/images", help='image_path')
parser.add_argument('--weights', type=str, default="results/exp1/SHHNet2_1_train2_results/best_model.pth")
parser.add_argument('--results_dir', type=str, default='results/test/predict.csv', help='results dir')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#----------------------------------------------------------------------------
model = opt.model(num_classes=6).to(device)
#----------------------------------------------------------------------------
model.load_state_dict(torch.load(opt.weights, map_location=device))
model.to(device)
model.eval()

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

root_folder_path = opt.path

correct_classes = {
    'paul-cezanne': 0,
    'pierre-auguste-renoir': 1,
    'camille-pissarro': 2,
    'vincent-van-gogh': 3,
    'claude-monet': 4,
    'edgar-degas': 5,
}


class_names = {v: k for k, v in correct_classes.items()}


csv_file = opt.results_dir
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['File Name', 'Predicted Class', 'Confidence'])


for root, dirs, files in os.walk(root_folder_path):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(root, filename)

            try:

                image = Image.open(file_path).convert('RGB')
                input_data = data_transform(image).unsqueeze(0)
                input_data = input_data.to(device)

                with torch.no_grad():
                    output = model(input_data)

                probabilities = torch.softmax(output, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                predicted_class = class_names[predicted_class_idx]
                confidence = probabilities[0, predicted_class_idx].item()

                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([filename, predicted_class, f'{confidence:.4f}'])

                print(f"File: {filename}")
                print(f"  Predicted Class: {predicted_class}")
                print(f"  Confidence: {confidence:.4f}\n")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

print(f"All results have been saved to {csv_file}")