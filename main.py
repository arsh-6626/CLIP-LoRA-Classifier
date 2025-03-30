import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from datasets.custom_dataset import custom_dataset
from utils import *
from run_utils import *
from lora import *

def custom_collate_fn(batch):
    images = []
    labels= []
    paths = []
    for img, label, path in batch:
        # Convert PIL image to tensor if it's not already a tensor
        if not torch.is_tensor(img):
            transform = transforms.ToTensor()
            img = transform(img)
        images.append(img)
        labels.append(label)
        paths.append(path)
    # Stack tensors
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels, paths

def main():
    # Load config file
    args = get_arguments()
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100
    
    # Prepare dataset
    print("Preparing dataset.")
    dataset = custom_dataset("/home/cha0s/CLiP_LoDA/", preprocess=preprocess)
    
    train_loader = None
    if not args.eval_only:
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    
    # Create DataLoader with custom collate function
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=8, 
        shuffle=True, 
        pin_memory=True, 
        collate_fn=custom_collate_fn
    )
    
    infer_lora(args, clip_model, dataset, train_loader)
    # train_lora(args, clip_model, logit_scale, dataset, train_loader)
if __name__ == '__main__':
    main()