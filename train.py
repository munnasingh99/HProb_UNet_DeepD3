import time
from random import randrange
import csv
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model import *




def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, writer, device='cpu', val_dataloader=None, start_time=None): 
    history = {
        'training_time(min)': None
    }
    log_dir = f"{args.output_dir}/{args.stamp}/logs"
    os.makedirs(log_dir, exist_ok=True)
    train_log_file = open(f"{log_dir}/train_log.csv", 'w', newline='')
    val_log_file = open(f"{log_dir}/val_log.csv", 'w', newline='')
    

    train_writer = csv.writer(train_log_file)
    val_writer = csv.writer(val_log_file)
    header = ['epoch', 'iteration', 'loss_per_pixel', 'reconstruction_per_pixel', 'kl_term_per_pixel']
    header += [f'kl_scale_{i}_per_pixel' for i in range(args.latent_num)]
    header += ['learning_rate']
    
    train_writer.writerow(header)
    val_writer.writerow(header)

    def record_history(epoch,idx, loss_dict, type='train'):
        prefix = 'Minibatch Training ' if type == 'train' else 'Mean Validation '

        loss_per_pixel = loss_dict['loss'].item() / args.pixels
        reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
        kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
        kl_per_pixel = [ loss_dict['kls'][v].item() / args.pixels for v in range(args.latent_num) ]

        # Total Loss
        _dict = {   'total': loss_per_pixel,
                    'kl term': kl_term_per_pixel, 
                    'reconstruction': reconstruction_per_pixel  }
        writer.add_scalars(prefix + 'Loss Curve', _dict, idx)

        # KL Term Decomposition
        _dict = { 'sum': sum(kl_per_pixel) }
        _dict.update( { 'scale {}'.format(v): kl_per_pixel[v] for v in range(args.latent_num) } )
        writer.add_scalars(prefix + 'Loss Curve (K-L)', _dict, idx)

        # Coefficients
        if type == 'train':
            if args.loss_type.lower() == 'elbo':
                writer.add_scalar('Beta', criterion.beta_scheduler.beta, idx)
            elif args.loss_type.lower() == 'geco':
                lamda = criterion.log_inv_function(criterion.log_lamda).item()
                writer.add_scalar('Lagrange Multiplier', lamda, idx)
                writer.add_scalar('Beta', 1/(lamda+1e-20), idx)

        log_row = [e+1, idx, loss_per_pixel, reconstruction_per_pixel, kl_term_per_pixel]
        log_row += kl_per_pixel
        log_row += [lr_scheduler.get_last_lr()[0]]
        
        train_writer.writerow(log_row)
        train_log_file.flush()


    val_images, val_truths = next(iter(val_dataloader))
    val_images, val_truths = val_images[:16], val_truths[0][:16]
    truth_grid = make_grid(val_truths, nrow=4, pad_value=val_truths.min().item())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(truth_grid[0])
    ax.set_axis_off()
    fig.tight_layout()
    writer.add_figure('Validation Images / Ground Truth', fig)
    val_images_selection = val_images.to(device)
    
    last_time_checkpoint = start_time
    for e in range(args.epochs):
        for mb, (images, truths) in enumerate(tqdm(dataloader)):
            idx = e*len(dataloader) + mb+1

            # Initialization
            criterion.train()
            model.train()
            model.zero_grad()
            images, truths = images.to(device), truths[0].to(device)

            # Train One Step
            
            ## Get Predictions and Prepare for Loss Calculation
            if args.rec_type.lower() == 'mse':
                preds, infodicts = model(images, truths)
                preds, infodict = preds[:,0], infodicts[0]

            truths = truths.squeeze(dim=1)


            ## Calculate Loss
            loss = criterion(preds, truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])


            ## Backpropagate
            loss.backward()             # Calculate Gradients
            optimizer.step()            # Update Weights
            

            ## Step Beta Scheduler
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()


            # Record Train History
            loss_dict = criterion.last_loss.copy()
            loss_dict.update( { 'kls': infodict['kls'] } )

            record_history(e, idx, loss_dict)
            
            
            # Validation
            if idx % args.val_period == 0 and val_dataloader is not None:
                criterion.eval()
                model.eval()

                # Show Sample Validation Images
                with torch.no_grad():
                    val_preds = model(val_images_selection)[0]
                    
                    out_grid = make_grid(val_preds, nrow=4, pad_value=val_preds.min().item())

                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(out_grid[0].cpu())
                    ax.set_axis_off()
                    fig.tight_layout()
                    writer.add_figure('Validation Images / Prediction', fig, idx)

                # Calculate Validation Loss
                mean_val_loss, mean_val_reconstruction_term, mean_val_kl_term = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
                mean_val_kl = torch.zeros(args.latent_num, device=device)

                with torch.no_grad():
                    for _, (val_images, val_truths) in enumerate(val_dataloader):
                        val_images, val_truths = val_images.to(device), val_truths[0].to(device)
                        
                        ## Get Predictions and Prepare for Loss Calculation
                        if args.rec_type.lower() == 'mse':
                            preds, infodicts = model(val_images, val_truths)
                            preds, infodict = preds[:,0], infodicts[0]

                        val_truths = val_truths.squeeze(dim=1)

                        ## Calculate Loss
                        loss = criterion(preds, val_truths, kls=infodict['kls'])


                        mean_val_loss += loss
                        mean_val_reconstruction_term += criterion.last_loss['reconstruction_term']
                        mean_val_kl_term += criterion.last_loss['kl_term']
                        mean_val_kl += infodict['kls']
                    
                    val_minibatches = len(val_dataloader)
                    mean_val_loss /= val_minibatches
                    mean_val_reconstruction_term /= val_minibatches
                    mean_val_kl_term /= val_minibatches
                    mean_val_kl /= val_minibatches


                # Record Validation History
                loss_dict = {
                    'loss': mean_val_loss,
                    'reconstruction_term': mean_val_reconstruction_term,
                    'kl_term': mean_val_kl_term,
                    'kls': mean_val_kl
                }
                record_history(e,idx, loss_dict, type='val')
        
        
        # Report Epoch Completion
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        # print('Epoch {}/{} done in {:.1f} minutes. \t\t\t\t Total time: {:.1f} minutes'.format(e+1, args.epochs, epoch_time, total_time))
        last_time_checkpoint = time_checkpoint
        
        # Save Model and Loss
        if (e+1) % args.save_period == 0 and (e+1) != args.epochs:
            torch.save(model, '{}/{}/model{}.pth'.format(args.output_dir, args.stamp, e+1))
            torch.save(criterion, '{}/{}/loss{}.pth'.format(args.output_dir, args.stamp, e+1))
        
        # Step Learning Rate
        writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], e)
        lr_scheduler.step()


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