import time
import os
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from combine_model import HPUNet  # Ensure your modified model.py is imported

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, device='cpu', val_dataloader=None, start_time=None):
    # Create output directory.
    save_dir = os.path.join(args.output_dir, args.stamp)
    os.makedirs(save_dir, exist_ok=True)

    # Obtain a fixed validation batch.
    val_images, val_truths = next(iter(val_dataloader))
    fixed_val_images = val_images[:5]  # (5, C, H, W)
    # Combine the two ground truth masks along channel dimension.
    # Assume val_truths is a tuple: (dendrite, spine); each of shape (B, H, W).
    dendrite = val_truths[0][:5].unsqueeze(1)  # shape (5, 1, H, W)
    spine    = val_truths[1][:5].unsqueeze(1)    # shape (5, 1, H, W)
    fixed_val_truths = torch.cat([dendrite, spine], dim=1)  # shape (5, 2, H, W)

    # Optionally, save grids for the fixed validation images and ground truths.
    #save_image(make_grid(fixed_val_images, nrow=5, pad_value=fixed_val_images.min().item()),
               #os.path.join(save_dir, "val_images_grid.png"))
    #save_image(make_grid(fixed_val_truths, nrow=5, pad_value=fixed_val_truths.min().item()),
               #os.path.join(save_dir, "val_truths_grid.png"))

    if start_time is None:
        start_time = time.time()

    for epoch in range(args.epochs):
        # Training loop for one epoch.
        for mb, (images, truths) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            idx = epoch * len(dataloader) + mb + 1
            model.train()
            optimizer.zero_grad()
            images = images.to(device)
            # Combine ground truth masks from the tuple.
            dendrite_batch = truths[0].to(device).unsqueeze(1)  # (B, 1, H, W)
            spine_batch    = truths[1].to(device).unsqueeze(1)    # (B, 1, H, W)
            combined_truth = torch.cat([dendrite_batch, spine_batch], dim=1)  # (B, 2, H, W)

            # Forward pass: use times=1 for training.
            preds, infodicts = model(images, combined_truth, times=1, first_channel_only=False)
            preds = preds.squeeze(1)  # Now shape is (B, 2, H, W)

            loss = criterion(preds, combined_truth, kls=infodicts[0]['kls'], lr=lr_scheduler.get_last_lr()[0])
            loss.backward()
            optimizer.step()
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()

        # End of epoch: run inference on fixed validation images with times=10.
        model.eval()
        with torch.no_grad():
            preds, _ = model(fixed_val_images.to(device), y=None, times=10, first_channel_only=False)
            # preds shape: (5, 10, 2, H, W)
            # For visualization, for each validation image, we’ll create a grid with:
            # [Original image, Ground truth, then 10 prediction samples (each prediction is 2-channel)]
            for i in range(fixed_val_images.size(0)):
                orig = fixed_val_images[i].cpu()  # (C, H, W)
                gt = fixed_val_truths[i].cpu()      # (2, H, W)
                # Get 10 prediction samples for image i.
                sample_preds = preds[i].cpu()         # (10, 2, H, W)
                # For visualization, we can either visualize each channel separately or combine them.
                # Here, we will create two grids: one for dendrite and one for spine.
                # Split prediction samples into two sets:
                pred_dendrites = sample_preds[:, 0].unsqueeze(1)  # (10, 1, H, W)
                pred_spines    = sample_preds[:, 1].unsqueeze(1)  # (10, 1, H, W)
                # Also split ground truth:
                gt_dendrite = gt[0].unsqueeze(0)  # (1, H, W)
                gt_spine    = gt[1].unsqueeze(0)  # (1, H, W)
                # Create a row for dendrites: [orig (if desired), ground truth, then 10 predictions]
                # Here we visualize the ground truth and predictions.
                row_dendrite = torch.cat([gt_dendrite, pred_dendrites], dim=0)  # (11, 1, H, W)
                grid_dendrite = make_grid(row_dendrite, nrow=11, pad_value=0)
                dendrite_path = os.path.join(save_dir, f"epoch_{epoch+1}_val_img_{i+1}_dendrite.png")
                save_image(grid_dendrite, dendrite_path)

                # Similarly for spines.
                row_spine = torch.cat([gt_spine, pred_spines], dim=0)
                grid_spine = make_grid(row_spine, nrow=11, pad_value=0)
                spine_path = os.path.join(save_dir, f"epoch_{epoch+1}_val_img_{i+1}_spine.png")
                save_image(grid_spine, spine_path)

                print(f"Saved validation grids for image {i+1} of epoch {epoch+1}")

        lr_scheduler.step()
        epoch_time = (time.time() - start_time) / 60
        if epoch % args.save_period == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.1f} minutes.")

    total_time = (time.time() - start_time) / 60
    print(f"Training completed in {total_time:.1f} minutes.")
    return {}