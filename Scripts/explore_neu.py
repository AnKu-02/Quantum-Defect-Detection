import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuration
DATA_DIR = "Data/NEU-DET"
BATCH_SIZE = 16
IMAGE_SIZE = 32

def load_dataset(data_dir, batch_size=16, image_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader

def visualize_batch(loader, class_names):
    images, labels = next(iter(loader))
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    
    for i in range(16):
        axs[i // 4, i % 4].imshow(images[i][0], cmap="gray")
        axs[i // 4, i % 4].set_title(class_names[labels[i]])
        axs[i // 4, i % 4].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    dataset, loader = load_dataset(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    class_names = dataset.classes
    print(f"Loaded {len(dataset)} images from {len(class_names)} classes:")
    print(class_names)
    
    visualize_batch(loader, class_names)

if __name__ == "__main__":
    main()
