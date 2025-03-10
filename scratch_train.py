import time
from random import randrange

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from scratch_model import *

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, writer, device='cpu', val_dataloader=None, start_time=None): 
    history = {
        'training_time(min)': None
    }

    if val_dataloader is not None:
        val_minibatches = len(val_dataloader)

    def record_history(idx, loss_dict, type='train'):
        prefix = 'Minibatch Training ' if type == 'train' else 'Mean Validation '
        
        # Combine KL terms from both branches (average them)
        combined_kls = loss_dict.get('kls', None)
        if combined_kls is None and ('kls_spine' in loss_dict and 'kls_dendrite' in loss_dict):
            # Compute element-wise average over latent scales
            combined_kls = [(loss_dict['kls_spine'][v] + loss_dict['kls_dendrite'][v]) / 2 for v in range(args.latent_num)]
        
        loss_per_pixel = loss_dict['loss'].item() / args.pixels
        reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
        kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
        kl_per_pixel = [ combined_kls[v].item() / args.pixels for v in range(args.latent_num) ]
        
        # Log total losses
        _dict = { 'total': loss_per_pixel,
                  'kl term': kl_term_per_pixel, 
                  'reconstruction': reconstruction_per_pixel }
        writer.add_scalars(prefix + 'Loss Curve', _dict, idx)
        
        # Log KL decomposition
        _dict = { 'sum': sum(kl_per_pixel) }
        _dict.update({ 'scale {}'.format(v): kl_per_pixel[v] for v in range(args.latent_num) })
        writer.add_scalars(prefix + 'Loss Curve (K-L)', _dict, idx)
        
        # Log coefficients (if using ELBO or GECO)
        if type == 'train':
            if args.loss_type.lower() == 'elbo':
                writer.add_scalar('Beta', criterion.beta_scheduler.beta, idx)
            elif args.loss_type.lower() == 'geco':
                lamda = criterion.log_inv_function(criterion.log_lamda).item()
                writer.add_scalar('Lagrange Multiplier', lamda, idx)
                writer.add_scalar('Beta', 1/(lamda+1e-20), idx)

    # Prepare a fixed batch of validation images and ground truths for visualization
    val_images, (val_truths_spine, val_truths_dendrite) = next(iter(val_dataloader))
    val_images = val_images[:16]
    val_truths_spine = val_truths_spine[:16]
    val_truths_dendrite = val_truths_dendrite[:16]
    truth_grid = make_grid(val_truths_spine, nrow=4, pad_value=val_truths_spine.min().item())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(truth_grid[0])
    ax.set_axis_off()
    fig.tight_layout()
    writer.add_figure('Validation Images / Ground Truth', fig)
    val_images_selection = val_images.to(device)
    
    last_time_checkpoint = start_time
    for e in range(args.epochs):
        for mb, (images, (truths_spine, truths_dendrite)) in enumerate(tqdm(dataloader)):
            idx = e * len(dataloader) + mb + 1

            # Set model and loss to train mode and zero gradients
            criterion.train()
            model.train()
            optimizer.zero_grad()  # Prefer optimizer.zero_grad() over model.zero_grad()
            
            images = images.to(device)
            truths_spine = truths_spine.to(device)
            truths_dendrite = truths_dendrite.to(device)
            # Concatenate ground truths for the posterior branch if needed.
            truths = torch.cat([truths_spine, truths_dendrite], dim=1)
            
            # Forward pass: get predictions and latent information
            if args.rec_type.lower() == 'mse':
                preds_spine, preds_dendrite, infodicts_s, infodicts_d = model(images, truths)
                #print(preds_spine.shape, preds_dendrite.shape)
                preds_spine, infodict_s = preds_spine[:, 0], infodicts_s[0]
                preds_dendrite, infodict_d = preds_dendrite[:, 0], infodicts_d[0]
            # print(images.shape, truths_spine.shape, truths_dendrite.shape)
            # print(preds_spine.shape, preds_dendrite.shape)
            # (Optional) Remove unnecessary squeeze if dimensions are already correct:
            # truths = truths.squeeze(dim=1)
            
            # Calculate losses for each branch
            spine_loss = criterion(preds_spine, truths_spine[:, 0], kls=infodict_s['kls'], lr=lr_scheduler.get_last_lr()[0])
            dendrite_loss = criterion(preds_dendrite, truths_dendrite[:, 0], kls=infodict_d['kls'], lr=lr_scheduler.get_last_lr()[0])
            loss = spine_loss + dendrite_loss

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            
            # Step beta scheduler if using ELBO loss
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()
            
            # Prepare loss dict for logging (combine KLs for logging purposes)
            loss_dict = criterion.last_loss.copy()
            loss_dict.update({
                'kls_spine': infodict_s['kls'],
                'kls_dendrite': infodict_d['kls'],
                'kls': [(infodict_s['kls'][v] + infodict_d['kls'][v]) / 2 for v in range(args.latent_num)]
            })
            record_history(idx, loss_dict)
            
            # Validation step
            if idx % args.val_period == 0 and val_dataloader is not None:
                criterion.eval()
                model.eval()
                
                # Generate predictions on the fixed validation batch
                with torch.no_grad():
                    # Unpack all outputs from the model
                    val_preds_spines, val_preds_dendrites, _, _ = model(val_images_selection)
                    out_grid = make_grid(val_preds_spines, nrow=4, pad_value=val_preds_spines.min().item())
                    
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(out_grid[0].cpu())
                    ax.set_axis_off()
                    fig.tight_layout()
                    writer.add_figure('Validation Images / Prediction', fig, idx)
                
                # Calculate validation losses
                mean_val_loss = torch.zeros(1, device=device)
                mean_val_reconstruction_term = torch.zeros(1, device=device)
                mean_val_kl_term = torch.zeros(1, device=device)
                mean_val_kl = torch.zeros(args.latent_num, device=device)
                
                with torch.no_grad():
                    for _, (val_images, (val_truths_spine, val_truths_dendrite)) in enumerate(val_dataloader):
                        val_images = val_images.to(device)
                        val_truths_spine = val_truths_spine.to(device)
                        val_truths_dendrite = val_truths_dendrite.to(device)
                        val_truths = torch.cat([val_truths_spine, val_truths_dendrite], dim=1)
                        
                        if args.rec_type.lower() == 'mse':
                            preds_spine, preds_dendrite, infodicts_s, infodicts_d = model(val_images, val_truths)
                            preds_spine, infodict_s = preds_spine[:, 0], infodicts_s[0]
                            preds_dendrite, infodict_d = preds_dendrite[:, 0], infodicts_d[0]
                        
                        spine_loss_val = criterion(preds_spine, val_truths_spine[:, 0], kls=infodict_s['kls'])
                        tmp_last_loss_spine = criterion.last_loss.copy()
                        dendrite_loss_val = criterion(preds_dendrite, val_truths_dendrite[:, 0], kls=infodict_d['kls'])
                        tmp_last_loss_dendrite = criterion.last_loss.copy()
                        
                        loss_val = spine_loss_val + dendrite_loss_val
                        
                        mean_val_loss += loss_val
                        # Average reconstruction and KL terms from both branches
                        mean_val_reconstruction_term += (tmp_last_loss_spine['reconstruction_term'] + tmp_last_loss_dendrite['reconstruction_term']) / 2
                        mean_val_kl_term += (tmp_last_loss_spine['kl_term'] + tmp_last_loss_dendrite['kl_term']) / 2
                        mean_val_kl += (infodict_s['kls'] + infodict_d['kls']) / 2
                    
                    mean_val_loss /= val_minibatches
                    mean_val_reconstruction_term /= (val_minibatches * 2)
                    mean_val_kl_term /= (val_minibatches * 2)
                    mean_val_kl /= (val_minibatches * 2)
                
                loss_dict_val = {
                    'loss': mean_val_loss,
                    'reconstruction_term': mean_val_reconstruction_term,
                    'kl_term': mean_val_kl_term,
                    'kls': mean_val_kl
                }
                record_history(idx, loss_dict_val, type='val')
        
        # Epoch completion logging
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        print(f"Epoch {e+1}/{args.epochs} completed in {epoch_time:.1f} minutes (Total: {total_time:.1f} minutes)")
        last_time_checkpoint = time_checkpoint
        
        # Save checkpoint periodically
        if (e+1) % args.save_period == 0 and (e+1) != args.epochs:
            torch.save(model, f"{args.output_dir}/{args.stamp}/model{e+1}.pth")
            torch.save(criterion, f"{args.output_dir}/{args.stamp}/loss{e+1}.pth")
        
        # Update learning rate scheduler
        writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], e)
        lr_scheduler.step()

    return history
