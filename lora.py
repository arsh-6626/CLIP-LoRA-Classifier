import torch
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
from utils import *
import cv2
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from tqdm import tqdm
import clip


def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples
    return acc

def evaluate_lora2(args, clip_model, loader, dataset, base_save_dir):
    base_save_dir = "/home/cha0s"
    clip_model.eval()
    
    # Ensure the base save directory exists
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Create class directories
    for text in dataset.classnames:
        os.makedirs(f"{base_save_dir}/{text}", exist_ok=True)
        
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    acc = 0.
    tot_samples = 0
    
    with torch.no_grad():
        for i, (images, _, paths) in enumerate(loader):
            images = images.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            predicted_classes = cosine_similarity.argmax(dim=-1)
            
            # Process each prediction
            for j, pred in enumerate(predicted_classes):
                pred_idx = pred.item()
                path = paths[j]
                print(path)
                prediction_text = f"For the highlighted limb {dataset.classnames[pred_idx]} is present."
                # Save the image to the appropriate class directory
                output_path = f"{base_save_dir}/{dataset.classnames[pred_idx]}/{path.split('/')[-1]}"
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        cv2.imwrite(output_path, img)
                    else:
                        print(f"Warning: Could not read image {path}")
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                print(prediction_text)
    
    return acc

def train_lora(args, clip_model, logit_scale, dataset, train_loader):
    VALIDATION = False
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    # Pre-load test features
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    # training LoRA
    scaler = torch.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            # if isinstance(target[0], str):  # Check if target contains strings
            #     target = [dataset.classnames.index(label) for label in target]  # Map strings to indices
            target = target.cuda()  # Ensure target is a Tensor and move it to GPU
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images = images.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            # scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            # current_lr = scheduler.get_last_lr()[0]
            print('Acc: {:.4f}, Loss: {:.4f}'.format( acc_train, loss_epoch))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def infer_lora(args, clip_model, dataset, train_loader):
    VALIDATION = False
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    print("\nLoading visual features and labels from val set.")
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    # Zero-shot CLIP
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.eval()
    clip_model = clip_model.cuda()
    acc_test = evaluate_lora2(args, clip_model, train_loader, dataset, "./")
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return