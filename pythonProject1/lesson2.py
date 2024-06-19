import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List


import torch
import torch.nn as nn
from sympy.physics.control.control_plots import matplotlib
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


words = open('names.txt', 'r').read().splitlines()
print(len(words))

b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1

#print(sorted(b.items(), key = lambda kv: -kv[1]))

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1
"""
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
#plt.show() activate to see the figure
#print(N[0])
"""
p = N[0].float()
p = p / p.sum()
#print(p)

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(itos[ix])

g = torch.Generator().manual_seed(2147483647)

P = (N+1).float()
P /= P.sum(1, keepdims=True)


for i in range(5):

  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))


log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

xs, ys = [],[]

for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)

  xs = torch.tensor(xs)
  ys = torch.tensor(ys)
print(xs)
print(ys)

xenc = F.one_hot(xs, num_classes=27).float()
print(xenc.shape)
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

logits = xenc @ W # log-counts
counts = logits.exp() # equivalent N
probs = counts / counts.sum(1, keepdims= True)


# forward pass
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(5), ys].log().mean()
print(loss.item())

# backward pass
W.grad = None # set to zero the gradient
loss.backward()

W.data += -0.1 * W.grad
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(5), ys].log().mean()



# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(150):
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float()  # input to the network: one-hot encoding
  logits = xenc @ W  # predict log-counts
  counts = logits.exp()  # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
  print(loss.item())

  # backward pass
  W.grad = None  # set to zero the gradient
  loss.backward()

  # update
  W.data += -20 * W.grad

  # finally, sample from the 'neural net' model
  g = torch.Generator().manual_seed(2147483647)




for i in range(5):

    out = []
    ix = 0
    while True:

      # ----------
      # BEFORE:
      # p = P[ix]
      # ----------
      # NOW:
      xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
      logits = xenc @ W  # predict log-counts
      counts = logits.exp()  # counts, equivalent to N
      p = counts / counts.sum(1, keepdims=True)  # probabilities for next character
      # ----------

      ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
      out.append(itos[ix])
      if ix == 0:
        break
    print(''.join(out))

"""
    for i in range(5):

      out = []
      ix = 0
      while True:

        # ----------
        # BEFORE:
        p = P[ix]
        # ----------
        # NOW:
        #xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        #logits = xenc @ W  # predict log-counts
        #counts = logits.exp()  # counts, equivalent to N
        #p = counts / counts.sum(1, keepdims=True)  # probabilities for next character
        # ----------

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
          break
      print(''.join(out))

"""