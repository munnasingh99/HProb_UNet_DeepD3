import time
import os
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import HPUNet  # Import your model
# Assume criterion, optimizer, lr_scheduler, and dataloader/val_dataloader are defined elsewhere

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, device='cpu', val_dataloader=None, start_time=None):
    # Create output directory based on args.output_dir and a unique stamp.
    save_dir = os.path.join(args.output_dir, args.stamp)
    os.makedirs(save_dir, exist_ok=True)

    # Obtain a fixed validation batch (here, first 5 images) for later evaluation.
    # We assume that the dataloader returns (images, truths) where truths is a tuple and the first element is the mask.
    val_images, val_truths = next(iter(val_dataloader))
    fixed_val_images = val_images[:5]      # shape: (5, C, H, W)
    fixed_val_truths = val_truths[0][:5]     # shape: (5, C, H, W)

    # Optionally, save the original validation images and ground truth grids.
    save_image(make_grid(fixed_val_images, nrow=5, pad_value=fixed_val_images.min().item()),
               os.path.join(save_dir, "val_images_grid.png"))
    save_image(make_grid(fixed_val_truths, nrow=5, pad_value=fixed_val_truths.min().item()),
               os.path.join(save_dir, "val_truths_grid.png"))

    if start_time is None:
        start_time = time.time()

    for epoch in range(args.epochs):
        # Training loop for one epoch with a progress bar.
        for mb, (images, truths) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            idx = epoch * len(dataloader) + mb + 1
            model.train()
            optimizer.zero_grad()
            images, truths = images.to(device), truths[0].to(device)
            #print(f"Batch {idx}: images shape: {images.shape}, truths shape: {truths.shape}")
            # Forward pass: for training we use one prediction (first sample)
            preds, infodicts = model(images, truths)
            preds, infodict = preds[:,0], infodicts[0]
            
            truths = truths.squeeze(dim=1)
            preds = preds.squeeze(dim=1)  # remove channel dimension if needed
            # Calculate loss (your criterion expects kls and learning rate as additional arguments)
            loss = criterion(preds, truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])
            loss.backward()
            optimizer.step()

            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()

        # End of epoch: run inference on fixed validation images with times=10 to sample prediction diversity.
        model.eval()
        with torch.no_grad():
            # Generate 10 predictions per fixed validation image
            # This returns a tensor of shape (5, 10, H, W) if first_channel_only=True.
            preds, _ = model(fixed_val_images.to(device), y=None, times=10)
            # Apply sigmoid if the model outputs logits.
            #preds = torch.sigmoid(preds)
            print(f"Predictions shape: {preds.shape}")  # Should be (5, 10, H, W)
            # For each of the 5 images, build and save a grid:
            for i in range(fixed_val_images.size(0)):
                # Get original input and ground truth (move to CPU for saving)
                orig = fixed_val_images[i]
                gt = fixed_val_truths[i]
                # Get the 10 prediction samples for this image.
                # preds[i] has shape (10, H, W); if needed, add channel dimension.
                sample_preds = preds[i].cpu()  # shape: (10, H, W)
                if sample_preds.dim() == 3:
                    sample_preds = sample_preds.unsqueeze(1)  # shape: (10, 1, H, W)
                
                print((orig.is_cuda, gt.is_cuda, sample_preds.is_cuda))
                # Build a row with 12 images: [Original, GT, 10 predictions]
                row_images = [orig, gt] + [sample_preds[j] for j in range(sample_preds.size(0))]
                row_tensor = torch.stack(row_images, dim=0)
                # Arrange them horizontally
                grid = make_grid(row_tensor, nrow=2, pad_value=0)
                grid_save_path = os.path.join(save_dir, f"epoch_{epoch+1}_val_img_{i+1}.png")
                save_image(grid, grid_save_path)
                print(f"Saved grid for validation image {i+1} of epoch {epoch+1} at {grid_save_path}")

        # Step learning rate scheduler
        lr_scheduler.step()
        epoch_time = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.1f} minutes.")

    total_time = (time.time() - start_time) / 60
    print(f"Training completed in {total_time:.1f} minutes.")
    return {}  # return history as needed