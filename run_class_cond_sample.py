import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math

from tqdm import tqdm

import numpy as np
import mcubes
import trimesh


from PIL import Image

from pathlib import Path

from timm.models import create_model

import utils
import modeling_prob, modeling_vqvae

def get_args():
    parser = argparse.ArgumentParser('script', add_help=False)
    parser.add_argument('--model_pth', type=str, required=True, help='checkpoint path of model')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    # Model parameters
    parser.add_argument('--model', default='class_encoder_55_512_1024_24_K1024', type=str, metavar='MODEL')

    parser.add_argument('--category', type=int, required=True, default=-1)

    parser.add_argument('--vqvae', default='vqvae_512_1024_2048', type=str, metavar='MODEL')

    parser.add_argument('--vqvae_pth', required=True, type=str, metavar='MODEL')

    return parser.parse_args()


def get_model(args):
    model = create_model(
        args.model,
    )

    return model


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    model.to(device)
    checkpoint = torch.load(args.model_pth, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    vqvae = create_model(args.vqvae)
    vqvae.to(device)
    checkpoint = torch.load(args.vqvae_pth, map_location='cpu')
    vqvae.load_state_dict(checkpoint['model'])
    vqvae.eval()

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    N = 50000
    
    if args.id == -1:
        ids = range(0, 55)
    else:
        ids = range(args.id, args.id+1)

    id = args.category
    with torch.no_grad():
        categories = torch.Tensor([id]).long().cuda()

        cond = model.class_enc(categories)

        x, y, z, latent = model.sample(cond)

        centers = torch.cat([x[:, :, None], y[:, :, None], z[:, :, None]], dim=2).float() / 255.0 * 2 - 1

        latent = vqvae.codebook.embedding(latent)
        
        logits = torch.cat([vqvae.decoder(latent, centers, grid[:, i*N:(i+1)*N])[0] for i in range(math.ceil(grid.shape[1]/N))], dim=1)

        volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
        verts, faces = mcubes.marching_cubes(volume, 0)

        verts *= gap
        verts -= 1

        m = trimesh.Trimesh(verts, faces)
        m.export('sample.obj')

if __name__ == '__main__':
    opts = get_args()
    main(opts)
