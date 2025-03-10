import time
import os
import csv
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import *

def calculate_iou(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) for binary segmentation.
    
    Args:
        pred: Predicted logits (batch_size, h, w)
        target: Target binary mask (batch_size, h, w)
        threshold: Threshold for binary prediction
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        iou: Mean IoU score across the batch
    """
    pred_probs = torch.sigmoid(pred)
    pred_binary = (pred_probs > threshold).float()
    batch_size = pred.size(0)
    pred_binary = pred_binary.view(batch_size, -1)
    target = target.view(batch_size, -1)
    intersection = (pred_binary * target).sum(dim=1)
    union = pred_binary.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, device='cpu', val_dataloader=None, start_time=None): 
    """
    Training function for HPU-Net with CSV logging, image saving, and optional online hard-negative mining.
    
    Args:
        args: Command line arguments (expects attributes such as epochs, output_dir, stamp, pixels, hard_negative_mining)
        model: The HPU-Net model
        dataloader: Training data loader
        criterion: Loss function (e.g., ELBO or GECO loss that wraps a reconstruction loss)
        optimizer: Optimizer for training
        lr_scheduler: Learning rate scheduler
        device: Device to train on (cuda or cpu)
        val_dataloader: Validation data loader
        start_time: Start time for timing
    
    Returns:
        history: Dictionary of training metrics
    """
    history = {
        'training_time(min)': None
    }

    # Create directories for logs and images
    log_dir = f"{args.output_dir}/{args.stamp}/logs"
    img_dir = f"{args.output_dir}/{args.stamp}/val_images"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Create CSV log files
    train_log_file = open(f"{log_dir}/train_log.csv", 'w', newline='')
    val_log_file = open(f"{log_dir}/val_log.csv", 'w', newline='')
    
    train_writer = csv.writer(train_log_file)
    val_writer = csv.writer(val_log_file)
    
    # Write headers
    header = ['epoch', 'iteration', 'loss_per_pixel', 'reconstruction_per_pixel', 'kl_term_per_pixel']
    header += [f'kl_scale_{i}_per_pixel' for i in range(args.latent_num)]
    header += ['spine_iou', 'learning_rate']
    
    train_writer.writerow(header)
    val_writer.writerow(header)
    
    # Get a batch of validation images for visualization.
    # Consistently use spine labels as the target.
    if val_dataloader is not None:
        val_images, val_labels = next(iter(val_dataloader))
        val_images = val_images[:32].to(device)
        # Use spine labels for visualization (assuming index 1 holds spine labels)
        val_spines = val_labels[1][:32].to(device)
        save_validation_images(val_images, val_spines, 0, img_dir, "ground_truth")
        val_images_selection = val_images.clone()

    last_time_checkpoint = start_time

    # Training loop
    for e in range(args.epochs):
        model.train()
        criterion.train()
        
        for mb, (images, (dend_labels, spine_labels)) in enumerate(tqdm(dataloader, desc=f"Epoch {e+1}/{args.epochs}")):
            idx = e * len(dataloader) + mb + 1
            optimizer.zero_grad()
            
            # Use spine_labels consistently as the training target
            images = images.to(device)
            truths = spine_labels.to(device)
            
            preds, infodicts = model(images, truths)
            # Assuming output shape is (batch, times, H, W); take the first sample
            preds, infodict = preds[:, 0], infodicts[0]
            truths = truths.squeeze(dim=1)
            
            # Compute the loss via the criterion
            loss = criterion(preds, truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])
            
            # ----- Optional Online Hard-Negative Mining -----
            # If enabled, recompute the reconstruction loss using only the worst 2% pixels per image.
            if hasattr(args, 'hard_negative_mining') and args.hard_negative_mining:
                # Obtain per-pixel reconstruction loss from the reconstruction loss wrapper.
                rec_loss = criterion.reconstruction_loss.last_loss['expanded_loss']
                batch_size = rec_loss.shape[0]
                rec_loss_flat = rec_loss.view(batch_size, -1)
                # Calculate number of pixels corresponding to the worst 2% per image.
                k = max(1, int(0.02 * rec_loss_flat.shape[1]))
                topk_vals, _ = torch.topk(rec_loss_flat, k, dim=1)
                rec_term_hard = topk_vals.mean()
                # Use the KL term as computed in the criterion
                kl_term = criterion.last_loss['kl_term']
                loss = rec_term_hard + kl_term
            # --------------------------------------------------

            loss.backward()
            optimizer.step()
            
            # Step Beta Scheduler for ELBO loss if applicable
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()
            
            # Log training metrics to CSV
            loss_dict = criterion.last_loss.copy()
            loss_dict.update({'kls': infodict['kls']})
            
            loss_per_pixel = loss_dict['loss'].item() / args.pixels
            reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
            kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
            kl_per_pixel = [loss_dict['kls'][v].item() / args.pixels for v in range(args.latent_num)]
            
            spine_iou = calculate_iou(preds, truths)
            
            log_row = [e+1, idx, loss_per_pixel, reconstruction_per_pixel, kl_term_per_pixel]
            log_row += kl_per_pixel
            log_row += [spine_iou, lr_scheduler.get_last_lr()[0]]
            
            train_writer.writerow(log_row)
            train_log_file.flush()
        
        # Validation phase
        if val_dataloader is not None:
            criterion.eval()
            model.eval()
            
            mean_val_loss = torch.zeros(1, device=device)
            mean_val_reconstruction_term = torch.zeros(1, device=device)
            mean_val_kl_term = torch.zeros(1, device=device)
            mean_val_kl = torch.zeros(args.latent_num, device=device)
            mean_spine_iou = 0.0
            
            with torch.no_grad():
                # Generate predictions on a selected validation batch for visualization.
                val_preds, _ = model(val_images_selection)
                val_preds = val_preds[:, 0]
                pred_probs = torch.sigmoid(val_preds)
                # Ensure correct shape for visualization.
                if pred_probs.dim() == 3:
                    pred_probs = pred_probs.unsqueeze(1)
                
                pred_binary = (pred_probs > 0.5).float()
                save_validation_images(val_images_selection, pred_binary, e+1, img_dir, "predictions")
                
                # Evaluate on the full validation set.
                for val_idx, (val_images, (val_dend_labels, val_spine_labels)) in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {e+1}/{args.epochs}")):
                    val_images = val_images.to(device)
                    # Consistently use spine labels.
                    val_truths = val_spine_labels.to(device)
                    val_preds, val_infodicts = model(val_images, val_truths)
                    val_pred, val_infodict = val_preds[:, 0], val_infodicts[0]
                    val_truths = val_truths.squeeze(dim=1)
                    
                    val_loss = criterion(val_pred, val_truths, kls=val_infodict['kls'])
                    
                    mean_val_loss += val_loss.item()
                    mean_val_reconstruction_term += criterion.last_loss['reconstruction_term'].item()
                    mean_val_kl_term += criterion.last_loss['kl_term'].item()
                    mean_val_kl += val_infodict['kls']
                    mean_spine_iou += calculate_iou(val_pred, val_truths)
                
                val_samples = len(val_dataloader)
                mean_val_loss /= val_samples
                mean_val_reconstruction_term /= val_samples
                mean_val_kl_term /= val_samples
                mean_val_kl /= val_samples
                mean_spine_iou /= val_samples
                
                val_loss_per_pixel = mean_val_loss / args.pixels
                val_reconstruction_per_pixel = mean_val_reconstruction_term / args.pixels
                val_kl_term_per_pixel = mean_val_kl_term / args.pixels
                val_kl_per_pixel = [mean_val_kl[v].item() / args.pixels for v in range(args.latent_num)]
                
                val_log_row = [e+1, idx, val_loss_per_pixel, val_reconstruction_per_pixel, val_kl_term_per_pixel]
                val_log_row += val_kl_per_pixel
                val_log_row += [mean_spine_iou, lr_scheduler.get_last_lr()[0]]
                
                val_writer.writerow(val_log_row)
                val_log_file.flush()
                
                print(f"Epoch {e+1}/{args.epochs}: Spine IoU: {mean_spine_iou:.4f}")
        
        # Timing and checkpointing
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        print(f"Epoch {e+1}/{args.epochs} completed in {epoch_time:.1f} minutes. Total time: {total_time:.1f} minutes")
        last_time_checkpoint = time_checkpoint
        
        if (e+1) % args.save_period == 0 or (e+1) == args.epochs:
            torch.save(model, f"{args.output_dir}/{args.stamp}/model_{e+1}.pth")
            torch.save(criterion, f"{args.output_dir}/{args.stamp}/loss_{e+1}.pth")
        
        lr_scheduler.step()
    
    train_log_file.close()
    val_log_file.close()
    
    history['training_time(min)'] = (time.time() - start_time) / 60
    return history

def save_validation_images(images, masks, epoch, save_dir, prefix):
    """
    Save validation images to disk.
    
    Args:
        images: Input images [B, C, H, W]
        masks: Segmentation masks [B, 1, H, W]
        epoch: Current epoch
        save_dir: Directory to save images
        prefix: Prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    img_grid = make_grid(images.cpu(), nrow=4, normalize=True)
    mask_grid = make_grid(masks.cpu(), nrow=4, normalize=True)
    
    save_image(img_grid, f"{save_dir}/{prefix}_images_epoch_{epoch}.png")
    save_image(mask_grid, f"{save_dir}/{prefix}_predictions_epoch_{epoch}.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input Images")
    axes[0].axis('off')
    
    axes[1].imshow(mask_grid.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Masks")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_combined_epoch_{epoch}.png", dpi=600, bbox_inches='tight')
    plt.close(fig)
