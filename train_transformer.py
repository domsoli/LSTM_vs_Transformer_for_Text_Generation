# -*- coding: utf-8 -*-
# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import time
import torch
import numpy as np
from torch import nn
from torch import cuda, device, save


def get_batch(source, i, bptt = 35):

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target



def repackage_hidden(h):

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)



def evaluate(net, loss_fn, data_source, ntokens, bptt = 35):

    # Turn on evaluation mode which disables dropout.
    net.eval()
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            inputs, targets = get_batch(data_source, i)
            inputs = inputs.long()
            targets = targets.long()
            output = net(inputs)
            output_flat = output.view(-1, ntokens)
            total_loss += len(inputs) * loss_fn(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)



def split_train_test(dataset, train_prc=0.8):

    # Define dataset length
    n = len(dataset)
    # Define number of training dataset indices
    m = round(train_prc * n)
    # Split datasets in two
    train_idx = np.random.choice(n, m)
    train = [dataset[i] for i in range(n) if i in train_idx]
    test = [dataset[i] for i in range(n) if i not in train_idx]
    return torch.cat(train), torch.cat(test)



def batchify(data, bsz):

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data



def train(net, optimizer, loss_fn, data, ntokens, clip, epoch, bptt = 35, log_interval = 100):

    # Turn on training mode which enables dropout.
    net.train()

    total_loss = 0.
    start_time = time.time()
    temp_loss = []

    for batch, i in enumerate(range(0, data.size(0) - 1, bptt)):
        inputs, targets = get_batch(data, i)
        inputs = inputs.long()
        targets = targets.long()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the net would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        output = net(inputs)

        loss = loss_fn(output.view(-1, ntokens), targets)
        loss.backward()

        optimizer.step()

        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch, len(data) // bptt,
                elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
            temp_loss.append(float(cur_loss))
    return float(np.mean(temp_loss))



if __name__ == "__main__":
