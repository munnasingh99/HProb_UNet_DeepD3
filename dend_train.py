import time
import os
import csv
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import *

def calculate_iou(pred, target, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) for binary segmentation.
    
    Args:
        pred: Predicted binary mask (B, 1, H, W)
        target: Target binary mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        iou: Mean IoU score across the batch
    """
    # Convert predictions to binary masks (0 or 1)
    pred_binary = (pred > 0.5).float()
    #print(pred_binary.shape, target.shape)
    # Calculate intersection and union
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Return mean IoU across the batch
    return iou.mean().item()

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, device='cpu', val_dataloader=None, start_time=None): 
    """
    Simplified training function for HPU-Net with CSV logging and image saving.
    
    Args:
        args: Command line arguments
        model: The HPU-Net model
        dataloader: Training data loader
        criterion: Loss function (ELBO or GECO)
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
    header += ['dendrite_iou', 'learning_rate']
    
    train_writer.writerow(header)
    val_writer.writerow(header)
    
    # Get a batch of validation images for visualization
    if val_dataloader is not None:
        val_images, val_labels = next(iter(val_dataloader))
        val_images = val_images[:16].to(device)
        
        # For dendrite and spine visualization
        val_dendrites, val_spines = val_labels[0][:16], val_labels[1][:16]
        
        # Save ground truth images
        save_validation_images(val_images, val_dendrites,0, img_dir, "ground_truth")

    last_time_checkpoint = start_time
    for e in range(args.epochs):
        model.train()
        criterion.train()
        
        # Training loop
        for mb, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {e+1}/{args.epochs}")):
            idx = e*len(dataloader) + mb + 1
            
            # Initialize
            model.zero_grad()
            
            # Move to device
            images = images.to(device)
            dendrite_masks, spine_masks = labels[0].to(device), labels[1].to(device)
            
            # Combine masks for the combined target
            # combined_masks = torch.cat([dendrite_masks, spine_masks], dim=1)
            
            # Forward pass
            preds, infodicts = model(images, dendrite_masks)
            preds, infodict = preds, infodicts[0]
            
            # Split predictions if needed
            # dendrite_preds = preds[:, 0:1]
            # spine_preds = preds[:, 1:2]
            #print(preds.shape, dendrite_masks.shape)
            # Calculate loss
            loss = criterion(preds, dendrite_masks, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            # Step Beta Scheduler for ELBO
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()
            
            # Log training metrics to CSV
            loss_dict = criterion.last_loss.copy()
            loss_dict.update({'kls': infodict['kls']})
            
            loss_per_pixel = loss_dict['loss'].item() / args.pixels
            reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
            kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
            kl_per_pixel = [loss_dict['kls'][v].item() / args.pixels for v in range(args.latent_num)]
            
            # Calculate training IoU for monitoring
            dendrite_iou = calculate_iou(preds, dendrite_masks)
            # spine_iou = calculate_iou(spine_preds, spine_masks) if spine_preds is not None else 0.0
            # mean_iou = (dendrite_iou + spine_iou) / 2.0 if spine_preds is not None else dendrite_iou
            
            log_row = [e+1, idx, loss_per_pixel, reconstruction_per_pixel, kl_term_per_pixel]
            log_row += kl_per_pixel
            log_row += [dendrite_iou,lr_scheduler.get_last_lr()[0]]
            
            train_writer.writerow(log_row)
            train_log_file.flush()
        
        # End of epoch - do validation
        if val_dataloader is not None:
            criterion.eval()
            model.eval()
            
            # Initialize metrics for validation
            mean_val_loss = 0.0
            mean_val_reconstruction_term = 0.0
            mean_val_kl_term = 0.0
            mean_val_kl = torch.zeros(args.latent_num, device=device)
            mean_dendrite_iou = 0.0
            # mean_spine_iou = 0.0
            
            with torch.no_grad():
                # Visualize predictions on the selected validation batch
                val_preds, val_infodicts = model(val_images)
                
                #dendrite_preds = val_preds[:, 0:1, 0]
                #spine_preds = val_preds[:, 1:2, 0] if val_preds.shape[1] > 1 else None
                
                # Save prediction images
                save_validation_images(val_images, val_preds, e+1, img_dir, "predictions")
                
                # Calculate validation metrics
                for val_idx, (val_images, val_labels) in enumerate(val_dataloader):
                    val_images = val_images.to(device)
                    val_dendrites, val_spines = val_labels[0].to(device), val_labels[1].to(device)
                    val_combined = torch.cat([val_dendrites, val_spines], dim=1)
                    
                    # Forward pass
                    val_preds, val_infodicts = model(val_images, val_dendrites)
                    val_pred, val_infodict = val_preds[:,0], val_infodicts[0]
                    
                    # Calculate loss
                    val_loss = criterion(val_pred, val_dendrites, kls=val_infodict['kls'])
                    
                    # Split predictions for evaluation
                    #val_dendrite_pred = val_pred[:, 0:1]
                    #val_spine_pred = val_pred[:, 1:2] if val_pred.shape[1] > 1 else None
                    
                    # Calculate IoU scores
                    dendrite_iou = calculate_iou(val_preds, val_dendrites)
                    #spine_iou = calculate_iou(val_spine_pred, val_spines) if val_spine_pred is not None else 0.0
                    
                    # Accumulate metrics
                    mean_val_loss += val_loss.item()
                    mean_val_reconstruction_term += criterion.last_loss['reconstruction_term'].item()
                    mean_val_kl_term += criterion.last_loss['kl_term'].item()
                    mean_val_kl += val_infodict['kls']
                    mean_dendrite_iou += dendrite_iou
                    #mean_spine_iou += spine_iou
                
                # Average metrics
                val_samples = len(val_dataloader)
                mean_val_loss /= val_samples
                mean_val_reconstruction_term /= val_samples
                mean_val_kl_term /= val_samples
                mean_val_kl /= val_samples
                mean_dendrite_iou /= val_samples
                #mean_spine_iou /= val_samples
                #mean_iou = (mean_dendrite_iou + mean_spine_iou) / 2.0 if val_pred.shape[1] > 1 else mean_dendrite_iou
                
                # Log validation metrics
                val_loss_per_pixel = mean_val_loss / args.pixels
                val_reconstruction_per_pixel = mean_val_reconstruction_term / args.pixels
                val_kl_term_per_pixel = mean_val_kl_term / args.pixels
                val_kl_per_pixel = [mean_val_kl[v].item() / args.pixels for v in range(args.latent_num)]
                
                val_log_row = [e+1, idx, val_loss_per_pixel, val_reconstruction_per_pixel, val_kl_term_per_pixel]
                val_log_row += val_kl_per_pixel
                val_log_row += [mean_dendrite_iou,lr_scheduler.get_last_lr()[0]]
                
                val_writer.writerow(val_log_row)
                val_log_file.flush()
                
                print(f"Epoch {e+1}/{args.epochs}, Val Loss: {val_loss_per_pixel:.6f}, Rec: {val_reconstruction_per_pixel:.6f}, KL: {val_kl_term_per_pixel:.6f}")
                print(f"IoU Scores: Dendrite: {mean_dendrite_iou:.4f}")
        
        # Timing and reporting
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        print(f"Epoch {e+1}/{args.epochs} completed in {epoch_time:.1f} minutes. Total time: {total_time:.1f} minutes")
        last_time_checkpoint = time_checkpoint
        
        # Save model and loss periodically
        if (e+1) % args.save_period == 0 or (e+1) == args.epochs:
            torch.save(model, f"{args.output_dir}/{args.stamp}/model_{e+1}.pth")
            torch.save(criterion, f"{args.output_dir}/{args.stamp}/loss_{e+1}.pth")
        
        # Step learning rate scheduler
        lr_scheduler.step()
    
    # Close log files
    train_log_file.close()
    val_log_file.close()
    
    # Record total training time
    history['training_time(min)'] = (time.time() - start_time) / 60
    
    return history


def save_validation_images(images, dendrite_masks, epoch, save_dir, prefix):
    """
    Save validation images to disk
    
    Args:
        images: Input images [B, C, H, W]
        dendrite_masks: Dendrite segmentation masks [B, 1, H, W]
        spine_masks: Spine segmentation masks [B, 1, H, W]
        epoch: Current epoch
        save_dir: Directory to save images
        prefix: Prefix for saved files
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    spine_masks = None
    # Create grids
    img_grid = make_grid(images.cpu(), nrow=4, normalize=True)
    dendrite_grid = make_grid(dendrite_masks.cpu(), nrow=4, normalize=True)
    
    # Save individual grids
    save_image(img_grid, f"{save_dir}/{prefix}_images_epoch_{epoch}.png")
    save_image(dendrite_grid, f"{save_dir}/{prefix}_dendrites_epoch_{epoch}.png")
    
    # Save spine grid if available
    if spine_masks is not None:
        spine_grid = make_grid(spine_masks.cpu(), nrow=4, normalize=True)
        save_image(spine_grid, f"{save_dir}/{prefix}_spines_epoch_{epoch}.png")
    
    # Create a combined visualization with Matplotlib
    fig, axes = plt.subplots(1, 3 if spine_masks is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input Images")
    axes[0].axis('off')
    
    axes[1].imshow(dendrite_grid.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Dendrite Masks")
    axes[1].axis('off')
    
    if spine_masks is not None:
        axes[2].imshow(spine_grid.permute(1, 2, 0).cpu().numpy())
        axes[2].set_title("Spine Masks")
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_combined_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)