import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torchvision import transforms
from PIL import Image
from modelC import build_densenet
import os

class_names = ['CALC', 'CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM', 'NORM']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_names)

model = build_densenet(num_classes).to(device)
model.load_state_dict(torch.load("Clasification.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            return class_names[predicted.item()]
            
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return "Error"