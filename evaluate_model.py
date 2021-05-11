import os
import json
import torch
import numpy as np
import sparseconvnet as scn
import argparse
import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D, proj3d
from module.utils.muon_track_dataset import MuonPoseLoader
from datetime import datetime
from ynet.models_losses.uresnet import UNet
from ynet.models_losses.yresnet import YNet, YNetLoss
from module.utils.torch_modules import SinusoidalPositionalEncoding
from ynet.models_losses.misc import fit_predict_torch, fit_predict_np, gaussian_kernel_np, gaussian_kernel_torch
from ynet.models_losses.metrics import purity, efficiency

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--weights_dir")
parser.add_argument("--epoch")
parser.add_argument("--n_per")
parser.add_argument("--max_seq_len")
parser.add_argument("--draw_predictions", type=bool, default=False)
args = parser.parse_args()

config = {
    "torch.manual_seed": 42,
    "n_batches": 5,
    "n_per": 4,
    "inner": {
        'num_classes': 3,
        'spatial_size': 128,
        'max_seq_len': 5
    },
    "ynet_config": {
        "full_chain_loss": {'seediness_weight': 1.0}
    },
    "n_epochs": 10,
    "log_dir": "full-sim/",
    "exp_name": "y_net_with_ysipm_big_dset_w_dropout_0.505-04-21_12:26:28",
}

torch.manual_seed(config['torch.manual_seed'])
device = 'cuda'

n_batches = config["n_batches"]
if args.n_per:
    n_per = int(args.n_per)
else:
    n_per = config["n_per"]
batch_size = n_batches * n_per

val = MuonPoseLoader("/home/jack/classes/thesis/data/val/", batch_size, device=device, return_energy=True)
inner = config['inner']

loss_fn = YNetLoss(config['ynet_config'])

if args.model.lower() == 'ynet' or args.model is None:
    model_cfg = {
        'ynet_full': inner,
        'yresnet_encoder':inner,
        'segmentation_decoder': inner,
        'seediness_decoder': inner,
        'embedding_decoder': inner,
        'network_base': {},
        'ppn': inner}
    model = YNet(model_cfg)
    model.to(device)

elif args.model.lower() == 'unet':
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
else:
    raise ValueError(f"Unrecognized model type {args.model}")

# set to evaluation mode
model.eval()

if args.weights_dir and args.epoch:
    if not os.path.isdir(args.weights_dir):
        raise OSError(f"Directory does not exist: {args.weights_dir}")
    if args.weights_dir[-1] != '/':
        args.weights_dir += '/'
    epochs = args.epoch + "epochs"
    weights = args.weights_dir + epochs
    print(f"\tLoading weights {weights}")
    model.load_state_dict(torch.load(weights))

if args.model.lower() == 'ynet' or args.model is None and args.max_seq_len:
    # silly hack because you can't load saved models directly,
    # they were trained with max_seq_len = 5 
    max_seq_len = int(args.max_seq_len)
    model.pe = SinusoidalPositionalEncoding(max_seq_len, 1).to(device)

if args.draw_predictions:
    # plot some events
    x_range = [-50, 50]
    y_range = [-150, 150]
    z_range = [-50, 50]
    EVENT_WRITE_DIR = './'
    def scatter3d_module(x,y,z, cs, tag, title=None, vtx=None, sym_z=True, colorsMap='jet', savegif=False):
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(x[1:-1], y[1:-1], z[1:-1], s=1, c=scalarMap.to_rgba(cs)[1:-1],)

        start = np.rint(np.array([x[0], y[0], z[0]]) * 2) / 2
        end = np.rint(np.array([x[-1], y[-1], z[-1]]) * 2) / 2

        # if vtx is None:
        #     ax.text(x[0], y[0], z[0], f'Start: {tuple(start)}', size=10, zorder=1, color='k')
        #     ax.text(x[-1], y[-1], z[-1], f'End: {tuple(end)}', size=10, zorder=1, color='k')
        #     ax.scatter(x[0], y[0], z[0], s=3, c='red',)
        #     ax.scatter(x[-1], y[-1], z[-1], s=3, c='red',)
        # else:
        #     ax.text(vtx[0], vtx[1], vtx[2], f'Vertex: {tuple(vtx.astype(int))}', size=10, zorder=1, color='k')
        #     ax.scatter(x[0], y[0], z[0], s=3, c='red',)

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_range[0], z_range[1])

        ax.set_xlabel('X[cm]')
        ax.set_ylabel('Y[cm]')
        ax.set_zlabel('Z[cm]')
        if title:
            ax.set_title(title)
        scalarMap.set_array(cs)

        if not savegif:
            plt.savefig(EVENT_WRITE_DIR + f"event{tag}.png")
            plt.close()
            return

        def animate(frame):
            ax.view_init(30, frame/4)
            return fig

        anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)
        anim.save(EVENT_WRITE_DIR + f"event{tag}.gif", writer=animation.PillowWriter(fps=10))
        plt.close()

    events = [0, 1, 2, 3]
    for i in events:
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

        with torch.no_grad():
            if isinstance(model, YNet):
                predictions = model(primary, secondary)
            else:
                predictions = model(primary)

        pred_labels, _ = fit_predict_torch(
            predictions['embeddings'][0],
            predictions['seediness'][0],
            predictions['margins'][0],
            gaussian_kernel_torch
        )
        energy[0] = (energy[0].cpu().numpy() * np.array([100/128, 300/128, 100/128, 1])) - np.array([50, 150, 50, 0])
        pred_labels = pred_labels.cpu().numpy()
        fragment_labels = fragment_labels.squeeze().cpu().numpy()

        scatter3d_module(energy[0][:, 0], energy[0][:, 1], energy[0][:, 2], fragment_labels, f'true{i}')
        #scatter3d_module(energy[0][:, 0], energy[0][:, 1], energy[0][:, 2], fragment_labels, f'true{i}', title=f'true event {i}', savegif=True)

        scatter3d_module(energy[0][:, 0], energy[0][:, 1], energy[0][:, 2], pred_labels, f'pred{i}')
        #scatter3d_module(energy[0][:, 0], energy[0][:, 1], energy[0][:, 2], pred_labels, f'pred{i}', title=f'predicted event {i}', savegif=True)

print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')

total_accuracy = []
total_purity = []
total_efficiency = []
for i in range(len(val)):
    light, energy, target = val[i]

    # batched event label indices
    event_labels = energy[0][:, None, -1] // n_batches
    
    # group batch items into smaller inputs
    light[0][:, -1] %= n_batches
    energy[0][:, -1] %= n_batches
    target[0][:, -1] %= n_batches

    primary = [torch.hstack(energy)]
    secondary = [torch.hstack(light)]

    # can edit this later if you want 
    segmentation_labels = torch.zeros_like(primary[0][:, None, -1])
    target = torch.hstack((primary[0], event_labels, segmentation_labels))
    target = [target]

    with torch.no_grad():
        if isinstance(model, YNet):
            predictions = model(primary, secondary)
        else:
            predictions = model(primary)

    res = loss_fn(predictions, target)
    loss, accuracy = res['loss'], res['accuracy']

    '''
    pred_labels, _ = fit_predict_torch(
        predictions['embeddings'][0],
        predictions['seediness'][0],
        predictions['margins'][0],
        gaussian_kernel_torch
    )
    '''
    pred_labels, _ = fit_predict_np(
        predictions['embeddings'][0].detach().cpu().numpy(),
        predictions['seediness'][0].detach().cpu().numpy(),
        predictions['margins'][0].detach().cpu().numpy(),
        gaussian_kernel_np
    )

    eff, pur = 0, 0
    for batch in range(n_batches):
        batch_idx = (primary[0][:, -2] == batch).detach().cpu().numpy()
        pred = pred_labels[batch_idx]
        label = event_labels.squeeze()[batch_idx].detach().cpu().numpy()
        e, p = efficiency(pred, label), purity(pred, label)
        eff += e
        pur += p
    eff /= n_batches
    pur /= n_batches

    total_efficiency.append(eff)
    total_purity.append(pur)

    print(f'Iter {i}: {100*i/len(val):.3f}%: {100*accuracy:.3f}%')
print()

print(f"Average validation accuracy: {100*sum(total_accuracy)/len(total_accuracy):3f}%")
print(f"Average validation efficiency: {100*sum(total_efficiency)/len(total_efficiency):3f}%")
print(f"Average validation purity: {100*sum(total_purity)/len(total_purity):3f}%")