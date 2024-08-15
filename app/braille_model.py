import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import logging

class BrailleDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.classes = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image

# Transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((70, 1000)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def collate_fn(batch):
    images = [item for item in batch]
    images = torch.stack(images, 0)
    return images

class BrailleCNN(nn.Module):
    def __init__(self, num_classes, max_label_length):
        super(BrailleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((35, 475))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 35 * 475, 128)
        self.fc2 = nn.Linear(128, num_classes * max_label_length)
        self.num_classes = num_classes
        self.max_label_length = max_label_length

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 35 * 475)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.max_label_length, self.num_classes)  # Reshape for sequence output
        return x

def load_model(model_path, num_classes, max_label_length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrailleCNN(num_classes=num_classes, max_label_length=max_label_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def text_to_braille_image(text, braille_image_folder):
    max_length = 20
    char_height = 70
    char_width = 50
    total_width = min(len(text), max_length) * char_width

    combined_image = Image.new('RGB', (total_width, char_height), (255, 255, 255))

    for i, char in enumerate(text[:max_length]):
        if char == ' ':
            char = 'space'
        char_image_path = os.path.join(braille_image_folder, f"{char}.png")
        if os.path.exists(char_image_path):
            char_image = Image.open(char_image_path).convert('RGB')
            combined_image.paste(char_image, (i * char_width, 0))
        else:
            print(f"Warning: Image for character '{char}' not found.")

    return combined_image

def process_image(filename, new_filename, braille_image_folder):
    image = Image.open(filename).convert('RGB')

    # Check if the width is less than 1000px and add space.png if necessary
    if image.width < 1000:
        space_img = Image.open(os.path.join(braille_image_folder, 'space.png')).convert('RGB')
        new_image = Image.new('RGB', (1000, image.height), (255, 255, 255))
        new_image.paste(image, (0, 0))

        current_width = image.width
        while current_width < 1000:
            new_image.paste(space_img, (current_width, 0))
            current_width += space_img.width

        new_image.save(new_filename)
    else:
        image.save(new_filename)

# Function to transform DataLoader into batches of (image, label)
def load_data(data_loader):
    images = next(iter(data_loader))
    return images

# Function to transform model predictions into text
def predict_text(model, images):
    device = next(model.parameters()).device  # Get the device of the model
    images = images.to(device)  # Move the images to the same device as the model
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 2)
    
    return predicted

# Function to decode predictions
def decode_text(predictions, num_classes):
    decoded = []
    for pred in predictions:
        text = ''.join([chr(p + 97) if p != -1 else '' for p in pred])
        text = text.replace('{', '').replace('}', '')  # Removing unwanted braces
        text = text.strip()  # Trim any leading or trailing spaces
        decoded.append(text)
    return decoded

# Function to fit and predict using BrailleCNN
def fit_predict(model, data_loader):
    logging.info("Starting the prediction process in the pipeline.")
    images = load_data(data_loader)
    predictions = predict_text(model, images)
    decoded_predictions = decode_text(predictions, model.num_classes)
    logging.info("Prediction process in the pipeline completed.")
    return decoded_predictions

# Define the pipeline
def create_pipeline(model):
    logging.info("Initializing the pipeline.")
    pipeline = Pipeline([
        ('data_loader', FunctionTransformer(load_data, validate=False)),
        ('model_predict', FunctionTransformer(lambda x: fit_predict(model, x), validate=False))
    ])
    logging.info("Pipeline initialized.")
    return pipeline
