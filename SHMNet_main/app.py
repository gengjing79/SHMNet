from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.SHMNet import SHMNet2_1

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

try:
    model = SHMNet2_1(num_classes=6).to(device)
    model.load_state_dict(torch.load('results/0/SHMNet2_1_train2_results/best_model.pth', map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

correct_classes = {
    'paul-cezanne': 0,
    'pierre-auguste-renoir': 1,
    'camille-pissarro': 2,
    'vincent-van-gogh': 3,
    'claude-monet': 4,
    'edgar-degas': 5,
}

@app.route('/classify', methods=['POST'])
def classify():
    try:
        file = request.files['image']
        img = Image.open(file).convert('RGB')

        img_tensor = transform(img).unsqueeze(0)

        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        predicted_class_name = None
        for artist, idx in correct_classes.items():
            if idx == predicted.item():
                predicted_class_name = artist
                break

        return jsonify({'class': predicted_class_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return send_from_directory('html','index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.root_path, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)