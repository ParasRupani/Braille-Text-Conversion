import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import os

class BrailleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 1].split("/")[-1])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 0]
        label = torch.tensor([self.classes[char] for char in label], dtype=torch.long)  # Convert label to tensor
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((70, 950)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Use -1 as padding value
    return images, labels

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
    max_length = 19
    char_height = 70
    char_width = 50
    total_width = min(len(text), max_length) * char_width

    combined_image = Image.new('RGB', (total_width, char_height), (255, 255, 255))

    for i, char in enumerate(text[:max_length]):
        if char == ' ':
            char = ' '
        char_image_path = os.path.join(braille_image_folder, f"{char}.png")
        if os.path.exists(char_image_path):
            char_image = Image.open(char_image_path).convert('RGB')
            combined_image.paste(char_image, (i * char_width, 0))
        else:
            print(f"Warning: Image for character '{char}' not found.")

    return combined_image
