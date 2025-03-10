
"""
Training script for Hierarchical Probabilistic U-Net (HPUNet) 
for spine and dendrite segmentation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import datetime
import time

# Import model and data generator
from scratch_model import HPUNet, GECOLoss, MSELossWrapper
from datagen import DataGeneratorDataset

# Parse arguments
parser = argparse.ArgumentParser(description='Train HPUNet for spine and dendrite segmentation')
parser.add_argument('--train_data_path', type=str, default=r"dataset/DeepD3_Training.d3set", help='Path to the trainingdata file')
parser.add_argument('--val_data_path', type=str, default=r"dataset/DeepD3_Validation.d3set", help='Path to the validation data file')
parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save outputs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
parser.add_argument('--resume', type=str, default=None, help='Path to resume from checkpoint')
parser.add_argument('--samples_per_epoch', type=int, default=4096, help='Number of samples per epoch')
parser.add_argument('--val_samples', type=int, default=1024, help='Number of validation samples')
parser.add_argument('--image_size', type=int, default=128, help='Image size')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

# Model specific arguments
parser.add_argument('--base_channels', type=int, default=32, help='Base number of channels')
parser.add_argument('--latent_num', type=int, default=3, help='Number of latent scales')
parser.add_argument('--kappa', type=float, default=0.05, help='GECO kappa (reconstruction target)')

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_coefficient(pred, target, smooth=1e-5):
    """Calculate Dice coefficient"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def save_checkpoint(model, optimizer, epoch, loss, dice_score, args, is_best=False):
    """Save model checkpoint"""
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    filename = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dice_score': dice_score,
        'args': args
    }
    
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint at epoch {epoch}")
    
    if is_best:
        best_filename = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
        torch.save(checkpoint, best_filename)
        print(f"Saved best model with dice score: {dice_score:.4f}")

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    dice_score = checkpoint.get('dice_score', 0.0)
    return start_epoch, dice_score

def visualize_results(image, dendrite_target, spine_target, dendrite_pred, spine_pred, epoch, step, args):
    """Visualize and save prediction results"""
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Input image
    plt.subplot(2, 3, 1)
    plt.title('Input Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Dendrite target
    plt.subplot(2, 3, 2)
    plt.title('Dendrite Target')
    plt.imshow(dendrite_target, cmap='viridis')
    plt.axis('off')
    
    # Spine target
    plt.subplot(2, 3, 3)
    plt.title('Spine Target')
    plt.imshow(spine_target, cmap='viridis')
    plt.axis('off')
    
    # Input image (copy)
    plt.subplot(2, 3, 4)
    plt.title('Input Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Dendrite prediction
    plt.subplot(2, 3, 5)
    plt.title('Dendrite Prediction')
    plt.imshow(dendrite_pred, cmap='viridis')
    plt.axis('off')
    
    # Spine prediction
    plt.subplot(2, 3, 6)
    plt.title('Spine Prediction')
    plt.imshow(spine_pred, cmap='viridis')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'visualizations', f'vis_epoch_{epoch}_step_{step}.png'))
    plt.close()

def main():
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize datasets and data loaders
    train_dataset = DataGeneratorDataset(
        fn=args.train_data_path,
        samples_per_epoch=args.samples_per_epoch,
        size=(1,args.image_size, args.image_size),
        augment=True,
        shuffle=True,
    )
    
    val_dataset = DataGeneratorDataset(
        fn=args.val_data_path,
        samples_per_epoch=args.val_samples,
        size=(1,args.image_size, args.image_size),
        augment=False,
        shuffle=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    chs = [args.base_channels, args.base_channels*2, args.base_channels*4, args.base_channels*8]
    scale_depth = [3, 3, 3, 3]  # Number of res blocks per scale
    kernel_size = [3, 3, 3, 3]
    dilation = [1, 1, 1, 1]
    
    model = HPUNet(
        in_ch=1,  # Single channel input (grayscale images)
        chs=chs,
        latent_num=args.latent_num,
        out_ch=1,  # Single channel output for each decoder
        activation="ReLU",
        scale_depth=scale_depth,
        kernel_size=kernel_size,
        dilation=dilation,
        padding_mode='circular',
        latent_channels=[1 for _ in range(args.latent_num)],  # Scalar latents
        latent_locks=None,  # Default: all latents are sampled
        conv_dim=2  # 2D convolutions
    )
    
    model.to(device)
    print(f"Model initialized with {args.latent_num} latent scales")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize loss function
    reconstruction_loss = MSELossWrapper()
    loss_function = GECOLoss(
        reconstruction_loss=reconstruction_loss,
        kappa=args.kappa,
        decay=0.9,
        update_rate=0.01,
        device=device,
        log_inv_function='exp',
        conv_dim=2
    )
    
    # Initialize tracking variables
    start_epoch = 0
    best_dice_score = 0.0
    
    # Load checkpoint if resuming training
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        start_epoch, best_dice_score = load_checkpoint(args.resume, model, optimizer, device)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_dendrite_dice = 0.0
        train_spine_dice = 0.0
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_bar = tqdm(train_loader, desc="Training")
        
        for step, (images, targets) in enumerate(train_bar):
            # Move data to device
            images = images.to(device)
            dendrite_targets = targets[0].to(device)
            spine_targets = targets[1].to(device)
            
            # Combine targets for posterior
            combined_target = torch.cat([dendrite_targets, spine_targets], dim=1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass using posterior for training
            spine_output, dendrite_output, spine_infodicts, dendrite_infodicts = model(
                images, combined_target, times=1, first_channel_only=True
            )
            
            # Get the first sample
            spine_sample = spine_output[:, 0]
            dendrite_sample = dendrite_output[:, 0]
            
            # Get KL terms
            spine_kls = spine_infodicts[0]['kls']
            dendrite_kls = dendrite_infodicts[0]['kls']
            
            # Calculate loss
            spine_loss = loss_function(spine_sample, spine_targets[:, 0], spine_kls, lr=args.lr)
            dendrite_loss = loss_function(dendrite_sample, dendrite_targets[:, 0], dendrite_kls, lr=args.lr)
            
            # Combined loss
            loss = spine_loss + dendrite_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                spine_pred = (spine_sample > 0.5).float()
                dendrite_pred = (dendrite_sample > 0.5).float()
                
                spine_dice = dice_coefficient(spine_pred, spine_targets[:, 0])
                dendrite_dice = dice_coefficient(dendrite_pred, dendrite_targets[:, 0])
            
            # Update metrics
            train_loss += loss.item()
            train_spine_dice += spine_dice.item()
            train_dendrite_dice += dendrite_dice.item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'spine_dice': spine_dice.item(),
                'dendrite_dice': dendrite_dice.item()
            })
            
            # Visualize results periodically
            if step % 100 == 0:
                # Get example from batch
                idx = 0
                vis_image = images[idx, 0].cpu().numpy()
                vis_dendrite_target = dendrite_targets[idx, 0].cpu().numpy()
                vis_spine_target = spine_targets[idx, 0].cpu().numpy()
                vis_dendrite_pred = dendrite_pred[idx].cpu().numpy()
                vis_spine_pred = spine_pred[idx].cpu().numpy()
                
                visualize_results(
                    vis_image, vis_dendrite_target, vis_spine_target,
                    vis_dendrite_pred, vis_spine_pred,
                    epoch, step, args
                )
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_spine_dice /= len(train_loader)
        train_dendrite_dice /= len(train_loader)
        train_avg_dice = (train_spine_dice + train_dendrite_dice) / 2
        
        print(f"Training - Loss: {train_loss:.4f}, "
              f"Dendrite Dice: {train_dendrite_dice:.4f}, "
              f"Spine Dice: {train_spine_dice:.4f}, "
              f"Avg Dice: {train_avg_dice:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dendrite_dice = 0.0
        val_spine_dice = 0.0
        
        val_bar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for step, (images, targets) in enumerate(val_bar):
                # Move data to device
                images = images.to(device)
                dendrite_targets = targets[0].to(device)
                spine_targets = targets[1].to(device)
                
                # Forward pass (sampling from prior during inference)
                spine_output, dendrite_output, _, _ = model(
                    images, times=8, first_channel_only=True
                )
                
                # Take the average of 8 samples for prediction
                spine_pred_avg = spine_output.mean(dim=1)
                dendrite_pred_avg = dendrite_output.mean(dim=1)
                
                # Calculate dice coefficient
                spine_pred_binary = (spine_pred_avg > 0.5).float()
                dendrite_pred_binary = (dendrite_pred_avg > 0.5).float()
                
                spine_dice = dice_coefficient(spine_pred_binary, spine_targets[:, 0])
                dendrite_dice = dice_coefficient(dendrite_pred_binary, dendrite_targets[:, 0])
                
                # Update metrics
                val_spine_dice += spine_dice.item()
                val_dendrite_dice += dendrite_dice.item()
                
                # Update progress bar
                val_bar.set_postfix({
                    'spine_dice': spine_dice.item(),
                    'dendrite_dice': dendrite_dice.item()
                })
                
                # Visualize the last batch
                if step == len(val_loader) - 1:
                    # Get example from batch
                    idx = 0
                    vis_image = images[idx, 0].cpu().numpy()
                    vis_dendrite_target = dendrite_targets[idx, 0].cpu().numpy()
                    vis_spine_target = spine_targets[idx, 0].cpu().numpy()
                    vis_dendrite_pred = dendrite_pred_binary[idx].cpu().numpy()
                    vis_spine_pred = spine_pred_binary[idx].cpu().numpy()
                    
                    visualize_results(
                        vis_image, vis_dendrite_target, vis_spine_target,
                        vis_dendrite_pred, vis_spine_pred,
                        epoch, step, args
                    )
        
        # Calculate validation metrics
        val_spine_dice /= len(val_loader)
        val_dendrite_dice /= len(val_loader)
        val_avg_dice = (val_spine_dice + val_dendrite_dice) / 2
        
        print(f"Validation - "
              f"Dendrite Dice: {val_dendrite_dice:.4f}, "
              f"Spine Dice: {val_spine_dice:.4f}, "
              f"Avg Dice: {val_avg_dice:.4f}")
        
        # Check if this is the best model
        is_best = val_avg_dice > best_dice_score
        if is_best:
            best_dice_score = val_avg_dice
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_avg_dice, args, is_best
            )
        
        # Create a training log
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Train Dice: {train_avg_dice:.4f}, "
                   f"Val Dice: {val_avg_dice:.4f}, "
                   f"Best Val Dice: {best_dice_score:.4f}\n")
    
    print(f"Training completed. Best validation Dice score: {best_dice_score:.4f}")

if __name__ == '__main__':
    main()