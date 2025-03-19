import argparse
import datetime
import json
import time
import socket
import tracemalloc
import os

import numpy as np
from torch.utils.data import DataLoader
from datagen import DataGeneratorDataset


from model import *
from analy_train import *

from torch.utils.tensorboard import SummaryWriter


# Define Arguments
parser = argparse.ArgumentParser(description="Hierarchical Probabilistic U-Net")

parser.add_argument("--random_seed", type=int, default=42, help="If provided, seed number will be set to the given value")


# Data

#parser.add_argument("--train_file", help="<Required> Path to the Training Set", required=True)
#parser.add_argument("--train_size", type=int, help="Training Set Size (If not provided, the whole examples in the traininig set will be used)")

#parser.add_argument("--val_file", help="Path to the Validation Set")
#parser.add_argument("--val_size", type=int, help="Validation Set Size (If not provided, the whole examples in the validation set will be used)")
parser.add_argument("--val_period", type=int, default=1024,help="# Steps Between Consecutive Validations")
parser.add_argument("--val_bs", type=int, default=32, help="Validation Batch Size")

parser.add_argument("--normalization", help="Normalization Type (None/standard/log_normal)")
parser.add_argument("--train_data_path",default=r"dataset/DeepD3_Training.d3set", help="Path to the Training Data")
parser.add_argument("--val_data_path", default=r"dataset/DeepD3_Validation.d3set", help="Path to the Validation Data")
parser.add_argument("--image_size", type=int, default=128, help="Size of the Images")

# Model

parser.add_argument("--in_ch", type=int, default=1, help="# Input Channels")
parser.add_argument("--out_ch", type=int, default=1, help="# Output Channels")
parser.add_argument("--intermediate_ch", type=int, nargs='+', help="<Required> Intermediate Channels", default=[32, 64, 128, 256])
parser.add_argument("--kernel_size", type=int, nargs='+', help="Kernel Size of the Convolutional Layers at Each Scale")
parser.add_argument("--scale_depth", type=int, nargs='+', default=[1], help="Number of Residual Blocks at Each Scale")
parser.add_argument("--dilation", type=int, nargs='+', default=[1], help="Dilation at Each Scale")
parser.add_argument("--padding_mode", default='zeros', help="Padding Mode in the Decoder's Convolutional Layers")

parser.add_argument("--latent_num", type=int, default=4, help="Number of Latent Scales (Setting to zero results in a deterministic U-Net)")
parser.add_argument("--latent_chs", type=int, nargs='+',  help="Number of Latent Channels at Each Latent Scale (Setting to None results in 1 channel per scale)")
parser.add_argument("--latent_locks", type=int, nargs='+',default= [None,None,None,None],help="Whether Latent Space in Locked at Each Latent Scale (Setting to None makes all scales unlocked)")


# Loss

parser.add_argument("--rec_type", help="Reconstruction Loss Type", default="mse")
parser.add_argument("--loss_type", default="ELBO", help="Loss Function Type (ELBO/GECO)")

parser.add_argument("--beta", type=float, default=1.0, help="(If Using ELBO Loss) Beta Parameter")
parser.add_argument("--beta_asc_steps", type=int, help="(If Using ELBO Loss with Beta Scheduler) Number of Ascending Steps (If Not Provided, Beta Will be Constant)")
parser.add_argument("--beta_cons_steps", type=int, default=1, help="(If Using ELBO Loss with Beta Scheduler) Number of Constant Steps")
parser.add_argument("--beta_saturation_step", type=int, help="(If Using ELBO Loss with Beta Scheduler) The Step at Which Beta Becomes Permanently 1")

parser.add_argument("--kappa", type=float, default=1.0, help="(If Using GECO Loss) Kappa Parameter")
parser.add_argument("--kappa_px", action='store_true', help="(If Using GECO Loss) Kappa Parameter Type (If true, Kappa should be provided per pixel)")
parser.add_argument("--decay", type=float, default=0.9, help="(If Using GECO Loss) EMA Decay Rate/Smoothing Factor")
parser.add_argument("--update_rate", type=float, default=0.01, help="(If Using GECO Loss) Lagrange Multiplier Update Rate")


# Training

parser.add_argument("--epochs", type=int, help="<Required> Number of Epochs", default=30)
parser.add_argument("--bs", type=int, help="<Required> Batch Size", default=32)
parser.add_argument("--samples_per_epoch", type=int, default=32768, help="Number of Samples per Epoch")
parser.add_argument("--val_samples", type=int, default=4096,help="Number of Samples in the Validation Set")
parser.add_argument("--num_workers", type=int, default=4, help="Number of Workers for Data Loading")
parser.add_argument("--optimizer", default="adam", help="Optimizer")
parser.add_argument("--wd", type=float, default=0.0, help="Weight Decay Parameter")

parser.add_argument("--lr", type=float, default=0.00001,help="<Required> (Initial) Learning Rate")
parser.add_argument("--scheduler_type", default='cons', help="Scheduler Type (cons/step/milestones)")
parser.add_argument("--scheduler_step_size", type=int, default=128, help="Learning Rate Scheduler Step Size (If type is step)")
parser.add_argument("--scheduler_milestones", type=int, nargs='+', help="Learning Rate Scheduler Milestones (If type is milestones)")
parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Learning Rate Scheduler Gamma")

parser.add_argument("--save_period", type=int, default=128, help="Number of Epochs Between Saving the Model")

parser.add_argument("--output_dir", default="test", help="Output Directory")
parser.add_argument("--comment", default="", help="Comment to be Included in the Stamp")


# New argument for config file
parser.add_argument("--config", type=str, help="Path to JSON config file")

# Parse the initial arguments
args = parser.parse_args()

# If a config file is provided, load it and update args
if args.config:
    with open(args.config, "r") as f:
        config_args = json.load(f)
    # Update the argparse namespace with values from the config file
    for key, value in config_args.items():
        setattr(args, key, value)



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

print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=args.bs,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)
    
val_loader = DataLoader(
    val_dataset,
    batch_size=args.val_bs,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

s = next(iter(train_dataset))[0].shape[-1]
args.size = s
args.pixels = s*s


if args.latent_locks is None:
    args.latent_locks = [0] * args.latent_num
args.latent_locks = [bool(l) for l in args.latent_locks]

if len(args.kernel_size) < len(args.intermediate_ch):
    if len(args.kernel_size) == 1:
        args.kernel_size = args.kernel_size * len(args.intermediate_ch)
    else:
        print('Invalid kernel size, exiting...')
        exit()

if len(args.dilation) < len(args.intermediate_ch):
    if len(args.dilation) == 1:
        args.dilation = args.dilation * len(args.intermediate_ch)
    else:
        print('Invalid dilation, exiting...')
        exit()

if len(args.scale_depth) < len(args.intermediate_ch):
    if len(args.scale_depth) == 1:
        args.scale_depth = args.scale_depth * len(args.intermediate_ch)
    else:
        print('Invalid scale depth, exiting...')
        exit()


# Set Random Seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
else:
    np.random.seed(0)
    torch.manual_seed(0)


# Set Device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.available_gpus = torch.cuda.device_count()
device = torch.device(args.device)
print("Device is {}".format(device))


# Generate Stamp
# stamp = 'My Lovely HPUnet'  # Assign a name manually
timestamp = datetime.datetime.now().strftime('%m%d-%H%M')  # Assign a timestamp
compute_node = socket.gethostname()
suffix = datetime.datetime.now().strftime('%f')
stamp = timestamp + '_' + compute_node[:2] + '_' + suffix + '_' + args.comment
print('Stamp:', stamp)
args.compute_node = compute_node
args.stamp = stamp


# Initialize SummaryWriter (for tensorboard)
writer = SummaryWriter('{}/{}/tb'.format(args.output_dir, stamp))


# # Load Data
# train_data = prepare_data(args.train_file, size=args.train_size, normalization=args.normalization)[0]
# train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
# s = next(iter(train_data))[0].shape[-1]
# args.size = s
# args.pixels = s*s

# val_file, val_loader = args.val_file, None
# if val_file is not None:
#     val_data = prepare_data(val_file, size=args.val_size, normalization=args.normalization)[0]
#     val_loader = DataLoader(val_data, batch_size=args.val_bs, shuffle=False)


# Initialize Model
model = HPUNet( in_ch=args.in_ch, out_ch=args.out_ch, chs=args.intermediate_ch,
                latent_num=args.latent_num, latent_channels=args.latent_chs, latent_locks=args.latent_locks,
                scale_depth=args.scale_depth, kernel_size=args.kernel_size, dilation=args.dilation,
                padding_mode=args.padding_mode )


args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model.to(device)


# Set Loss Function

## Reconstruction Loss
if args.rec_type.lower() == 'mse':
    reconstruction_loss = MSELossWrapper()

else:
    print('Invalid reconstruction loss type, exiting...')
    exit()

## Total Loss
if args.loss_type.lower() == 'elbo':
    if args.beta_asc_steps is None:
        beta_scheduler = BetaConstant(args.beta)
    else:
        beta_scheduler = BetaLinearScheduler(ascending_steps=args.beta_asc_steps, constant_steps=args.beta_cons_steps, max_beta=args.beta, saturation_step=args.beta_saturation_step)
    criterion = ELBOLoss(reconstruction_loss=reconstruction_loss, beta=beta_scheduler).to(device)

elif args.loss_type.lower() == 'geco':
    kappa = args.kappa
    if args.kappa_px is True:
        kappa *= args.pixels
    criterion = GECOLoss(reconstruction_loss=reconstruction_loss, kappa=kappa, decay=args.decay, update_rate=args.update_rate, device=device).to(device)

else:
    print('Invalid loss type, exiting...')
    exit()


# Set Optimizer
if args.optimizer == 'adamax':
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wd)

elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

else:
    print('Optimizer not known, exiting...')
    exit()


# Set LR Scheduler
if args.scheduler_type == 'cons':
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs)

elif args.scheduler_type == 'step':
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

elif args.scheduler_type == 'milestones':
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)


# Save Args
argsdict = vars(args)
with open('{}/{}/args.json'.format(args.output_dir, stamp), 'w') as f:
    json.dump(argsdict, f)
with open('{}/{}/args.txt'.format(args.output_dir, stamp), 'w') as f:
    for k in argsdict.keys():
        f.write("'{}': '{}'\n".format(k, argsdict[k]))


# Start Timing
start = time.time()

# Train the Model
history = train_model(args, model, train_loader, criterion, optimizer, lr_scheduler,device, val_loader, start)


# End Timing & Report Training Time
end = time.time()
training_time = (end - start) / 3600
history['training_time(hours)'] = training_time
print('Training done in {:.1f} hours'.format(training_time))


# Save Model, Loss and History
torch.save(model, '{}/{}/model.pth'.format(args.output_dir, stamp))

torch.save(criterion, '{}/{}/loss.pth'.format(args.output_dir, stamp))

with open('{}/{}/history.json'.format(args.output_dir, stamp), 'w') as f:
    json.dump(history, f)