import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

template = ["the photo resembles the node {}."]
classes = ["injury", "no_injury", "injury_and_amputation"]

class custom_dataset:
    def __init__(self, root_path, preprocess=None):
        self.classnames = classes
        self.root_path = root_path
        self.preprocess = preprocess
        self.train_x = datasets.ImageFolder(root_path, transform=None)  # No transform here
        self.template = template
    
    def __len__(self):
        return len(self.train_x)
    
    def __getitem__(self, idx):
        image_path, label = self.train_x.samples[idx]
        # Get the image path
        path = image_path
        # Load the image
        image = self.train_x.loader(path)
        # Apply transformations if needed
        if self.preprocess:
            image = self.preprocess(image)
        return image, label, image_path # Return the image and its path