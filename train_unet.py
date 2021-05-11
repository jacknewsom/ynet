import os
import json
import torch
from module.utils.muon_track_dataset import MuonPoseLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sparseconvnet as scn

config = {
    "torch.manual_seed": 42,
    "n_batches": 5,
    "n_per": 4,
    "inner": {
        'num_classes': 3,
        'spatial_size': 128,
        'max_seq_len': 5,
    },
    "ynet_config": {
        "full_chain_loss": {'seediness_weight': 1.0}
    },
    "n_epochs": 40,
    "log_dir": "full-sim/",
    "exp_name": "u_net_with_ysipm_big_dset",
}

exp_dir = datetime.now().strftime(f"{config['log_dir']}weights/u_net/{config['exp_name']}%m-%d-%y_%H:%M:%S/")

os.mkdir(exp_dir)

json.dump(config, open(exp_dir + "config.json", "w"))

torch.manual_seed(config['torch.manual_seed'])
device = 'cuda'

# number of batches per iteration
n_batches = config['n_batches']
# this is the actual batch size
n_per = config['n_per']
# this isn't the actual batch size, it's the number of events we need to load per iteration
batch_size = n_batches * n_per
train = MuonPoseLoader("/home/jack/classes/thesis/data/train/", batch_size, device=device, return_energy=True)
val = MuonPoseLoader("/home/jack/classes/thesis/data/val/", batch_size, device=device, return_energy=True)

from ynet.models_losses.uresnet import UNet
inner = config['inner']
model_cfg = {
    'unet_full': inner,
    'uresnet_encoder':inner,
    'segmentation_decoder': inner,
    'seediness_decoder': inner,
    'embedding_decoder': inner,
    'network_base': {},
    'ppn': inner}
model = UNet(model_cfg)
model.to(device)

from ynet.models_losses.yresnet import YNetLoss
loss_fn = YNetLoss(config['ynet_config'])

optim = torch.optim.AdamW(model.parameters())

n_epochs = config['n_epochs']

log_dir = config['log_dir']
exp_name = config['exp_name']

writer = SummaryWriter(log_dir=log_dir)
writer.add_scalar('batch_size', n_per)

if not os.path.isdir(log_dir + 'weights'):
    os.mkdir(log_dir + 'weights')

for epoch in range(n_epochs):
    print(f'Epoch {epoch}')
    model.train()
    train_loss = 0
    for i in range(len(train)):
        _, energy, target = train[i]

        # cluster labels
        fragment_labels = energy[0][:, None, -1] // n_batches
        # group batch items into smaller inputs
        energy[0][:, -1] %= n_batches
        target[0][:, -1] %= n_batches

        primary = [torch.hstack(energy)]

        # can edit this later if you want 
        segmentation_labels = torch.zeros_like(primary[0][:, None, -1])
        target = torch.hstack((primary[0], fragment_labels, segmentation_labels))
        target = [target]

        predictions = model(primary)

        res = loss_fn(predictions, target)
        loss, accuracy = res['loss'], res['accuracy']

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()

        if i % 10 == 0:
            print(f'\tIter {100*i/len(train):.3f}%: {loss.item()}', end='\r')
    train_loss /= len(train)
    print(f'\tAverage train loss: {train_loss}')

    torch.save(model.state_dict(), exp_dir + f"{epoch+1}epochs")

    model.eval()
    val_loss = 0
    for i in range(len(val)):
        _, energy, target = val[i]

        # cluster labels
        fragment_labels = energy[0][:, None, -1] // n_batches
        # group batch items into smaller inputs
        energy[0][:, -1] %= n_batches
        target[0][:, -1] %= n_batches

        primary = [torch.hstack(energy)]

        # can edit this later if you want 
        segmentation_labels = torch.zeros_like(primary[0][:, None, -1])
        target = torch.hstack((primary[0], fragment_labels, segmentation_labels))
        target = [target]

        with torch.no_grad():
            predictions = model(primary)

        res = loss_fn(predictions, target)
        loss, accuracy = res['loss'], res['accuracy']

        val_loss += loss.item()

        if i % 10 == 0:
            print(f'\tIter {100*i/len(val):.3f}%: {loss.item()}', end='\r')
    val_loss /= len(val)
    print(f'\tAverage validation loss: {val_loss}')
    writer.add_scalar('Loss/val', val_loss, epoch)

    torch.save(model.state_dict(), exp_dir + f"{epoch+1}epochs")