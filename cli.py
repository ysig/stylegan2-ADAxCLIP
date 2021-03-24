
import os
import sys
import random

import PIL
import matplotlib.pyplot as plt
import imageio
from tqdm.auto import trange

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF

def to_pil(img):
  return PIL.Image.fromarray(img, 'RGB')

def gen_random(dim):
  return torch.from_numpy(np.random.RandomState(0).randn(1, dim)).cuda()

def perpre(img):
  return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

class Stylegan2Gen(torch.nn.Module):
  def __init__(self, model_path, stylegan_dir, truncation_psi=0.5):
    super().__init__()
    sys.path.insert(1, stylegan_dir)
    import legacy
    with open(model_path, 'rb') as f:
      self.G = legacy.load_network_pkl(f)['G_ema'].cuda().eval()
    self.truncation = truncation_psi

  @property
  def z_dim(self):
    return self.G.z_dim

  def forward(self, z, truncation=None):
    label = torch.zeros([1, self.G.c_dim]).cuda()
    if truncation is None:
      truncation = self.truncation
    return self.G(z, label, truncation_psi=truncation)

  def gen_pil(self, z, truncation=None):
    with torch.no_grad():
      img = perpre(self.forward(z, truncation))
      return to_pil(img[0].cpu().numpy())

class Pars(torch.nn.Module):
    def __init__(self, dims, batch_size=16):
        super(Pars, self).__init__()
        self.z = torch.nn.Parameter(torch.zeros(batch_size, dims).normal_(std=1).cuda())

    def forward(self):
      return self.z

# Load the model

def plot(model, loss, iter, odir, lats):
  best = torch.topk(loss, k=1, largest=False)[1]
  if not os.path.isdir(odir):
    os.mkdir(odir)
  lts = lats()[best]
  model.gen_pil(lts.unsqueeze(0)).save(os.path.join(odir, f"{iter}.png"))
  return lts

def ascend_txt(model, perceptor, t, nom, lats, la, lb):
  out = model(lats(), 1.0)
  cutn, sideX, sideY = out.size()[1:]
  p_s = []
  for ch in range(cutn):
    size = int(sideX*torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
    offsetx = torch.randint(0, sideX - size, ())
    offsety = torch.randint(0, sideY - size, ())
    apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
    apper = torch.nn.functional.interpolate(apper, (224,224), mode='nearest')
    p_s.append(apper)
  into = torch.cat(p_s, 0)
  into = nom((into + 1) / 2)
  iii = perceptor.encode_image(into)

  llls = lats()
  lat_l = torch.abs(1 - torch.std(llls, dim=1)).mean() + torch.abs(torch.mean(llls)).mean() + 4*torch.clamp_max(torch.square(llls).mean(), 1)

  for array in llls:
    mean = torch.mean(array)
    diffs = array - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    lat_l = lat_l + torch.abs(kurtoses) / llls.shape[0] + torch.abs(skews) / llls.shape[0]
  
  return la*lat_l, -lb*torch.cosine_similarity(t, iii, dim=-1).mean()

def train(i, odir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb):
  optimizer.zero_grad()
  a, b = ascend_txt(model, perceptor, t, nom, lats, la, lb)
  loss = a + b
  loss.backward()
  optimizer.step()

  if i % plot_every == 0:
    plot(model, b, i, odir, lats)

def final(odir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb):
  with torch.no_grad():
    np.save(os.path.join(odir, 'final'), plot(model, ascend_txt(model, perceptor, t, nom, lats, la, lb)[1], 'final', odir, lats).cpu().numpy())

def imagine(text, model_path, lr=.07, seed=0, num_epochs=200, total_plots=20, batch_size=16, outdir=None, stylegan2_dir="stylegan2-ada-pytorch", clip_dir="CLIP", la=1, lb=100, truncation_psi=0.5):
    sys.path.insert(1, clip_dir)
    import clip
    perceptor, preprocess = clip.load('ViT-B/32')
    model = Stylegan2Gen(model_path, stylegan2_dir, truncation_psi)
    im_shape = perpre(model(gen_random(model.z_dim)))[0].size()
    sideX, sideY, channels = im_shape

    torch.manual_seed(seed)
    lats = Pars(model.z_dim, batch_size).cuda()
    optimizer = torch.optim.Adam(lats.parameters(), lr)

    nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    tx = clip.tokenize(text)
    t = perceptor.encode_text(tx.cuda()).detach().clone()
    
    outdir = (text if outdir is None else outdir)
    plot_every = int(num_epochs/total_plots)
    for i in trange(num_epochs):
        train(i, outdir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb)
    final(outdir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='stylegan2ada-image')
    parser.add_argument('-i', '--text', default="Your cat looks like the devil!")
    parser.add_argument('-n', '--network', required=True)
    parser.add_argument('-e', '--num-epochs', default=200, type=int)
    parser.add_argument('-p', '--total-plots', default=20, type=int)
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--truncation_psi', default=0.5, type=float)
    parser.add_argument('--lr', default=0.07, type=float)
    parser.add_argument('--la', default=1, type=float, help='Loss-factor a')
    parser.add_argument('--lb', default=100, type=float, help='Loss-factor b')
    parser.add_argument('-s', '--stylegan2-dir', default="stylegan2-ada-pytorch")
    parser.add_argument('-c', '--clip-dir', default="CLIP")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()
    imagine(args.text, args.network, args.lr, args.seed, args.num_epochs, args.total_plots, args.batch_size, args.outdir, args.stylegan2_dir, args.clip_dir, args.la, args.lb, args.truncation_psi)
