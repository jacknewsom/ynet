import os
import torch
import json
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
        'dropout_prob': 0.5,
    },
    "ynet_config": {
        "full_chain_loss": {'seediness_weight': 1.0}
    },
    "n_epochs": 40,
    "log_dir": "full-sim/",
    "exp_name": "y_net_with_ysipm_big_dset_w_dropout_0.5",
    "use_weights": None, 
    "use_epoch": None,
}

if not config['use_weights']:
    exp_dir = datetime.now().strftime(f"{config['log_dir']}weights/y_net/{config['exp_name']}%m-%d-%y_%H:%M:%S/")
    os.mkdir(exp_dir)
    json.dump(config, open(exp_dir + "config.json", "w"))
else:
    exp_dir = f"{config['log_dir']}weights/y_net/{config['use_weights']}"


torch.manual_seed(config["torch.manual_seed"])
device = 'cuda'

# number of batches per iteration
n_batches = config["n_batches"]
# this is the actual batch size
n_per = config["n_per"]
# this isn't the actual batch size, it's the number of events we need to load per iteration
batch_size = n_batches * n_per
train = MuonPoseLoader("/home/jack/classes/thesis/data/train/", batch_size, device=device, return_energy=True)
val = MuonPoseLoader("/home/jack/classes/thesis/data/val/", batch_size, device=device, return_energy=True)

from ynet.models_losses.yresnet import YNet
inner = config['inner']
model_cfg = {
    'ynet_full': inner,
    'yresnet_encoder':inner,
    'seediness_decoder': inner,
    'embedding_decoder': inner,
    'network_base': {},
}
model = YNet(model_cfg)
model.to(device)

from ynet.models_losses.yresnet import YNetLoss
loss_fn = YNetLoss(config['ynet_config'])

optim = torch.optim.AdamW(model.parameters())

n_epochs = config["n_epochs"]

log_dir = config["log_dir"]
exp_name = config["exp_name"]

writer = SummaryWriter(log_dir=exp_dir)
writer.add_scalar('batch_size', n_per)

if not os.path.isdir(log_dir + 'weights'):
    os.mkdir(log_dir + 'weights')

if config['use_weights'] and config['use_epoch']:
    if exp_dir != '/':
        exp_dir += '/'
    epochs = str(config['use_epoch']) + "epochs"
    weights = exp_dir + epochs
    print(f"\tLoading weights {weights}")
    model.load_state_dict(torch.load(weights))

if config['use_epoch']:
    epochs_range = range(config['use_epoch'], n_epochs)
else:
    epochs_range = range(n_epochs)

for epoch in epochs_range:
    print(f'Epoch {epoch}')

    model.train()
    train_loss = []
    for i in range(len(train)):
        light, energy, target = train[i]

        # cluster labels
        fragment_labels = energy[0][:, None, -1] // n_batches
        # group batch items into smaller inputs
        light[0][:, -1] %= n_batches
        energy[0][:, -1] %= n_batches
        target[0][:, -1] %= n_batches

        primary = [torch.hstack(energy)]
        secondary = [torch.hstack(light)]

        # can edit this later if you want 
        segmentation_labels = torch.zeros_like(primary[0][:, None, -1])
        target = torch.hstack((primary[0], fragment_labels, segmentation_labels))
        target = [target]

        print(f'Iter {i} ({100*i/len(train):.3f}%):')
        predictions = model(primary, secondary)

        res = loss_fn(predictions, target)
        loss, accuracy = res['loss'], res['accuracy']

        print(f'\t{loss.item()}')
        if torch.isnan(loss):
            raise ValueError()

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += [loss.item()]

        #if i % 10 == 0:
        #    print(f'\tIter {100*i/len(train):.3f}%: {loss.item()}', end='\r')

    train_loss = sum(train_loss) / len(train_loss)
    print(f'\tAverage train loss: {train_loss}')
    writer.add_scalar('Loss/train', train_loss, epoch)
    torch.save(model.state_dict(), exp_dir + f"{epoch+1}epochs") 

    model.eval()
    val_loss = []
    for i in range(len(val)):
        light, energy, target = val[i]

        # cluster labels
        fragment_labels = energy[0][:, None, -1] // n_batches
        # group batch items into smaller inputs
        light[0][:, -1] %= n_batches
        energy[0][:, -1] %= n_batches
        target[0][:, -1] %= n_batches

        primary = [torch.hstack(energy)]
        secondary = [torch.hstack(light)]

        # can edit this later if you want 
        segmentation_labels = torch.zeros_like(primary[0][:, None, -1])
        target = torch.hstack((primary[0], fragment_labels, segmentation_labels))
        target = [target]

        print(f'Iter {i} ({100*i/len(val):.3f}%):')
        with torch.no_grad():
            predictions = model(primary, secondary)

        res = loss_fn(predictions, target)
        loss, accuracy = res['loss'], res['accuracy']

        print(f'\t{loss.item()}')
        if torch.isnan(loss):
            continue

        val_loss += [loss.item()]

    val_loss = sum(val_loss) / len(val_loss)
    print(f'\tAverage validation loss: {val_loss}')
    writer.add_scalar('Loss/val', val_loss, epoch)
