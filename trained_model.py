
# -*- coding: utf-8 -*-

import json
import torch
import argparse
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from dataset import Encoder
from network import Network

# Parameters
parser = argparse.ArgumentParser(description='Train the text generator network.')

parser.add_argument('--type', type=str, default="rnn",
                    help='Type of Network to load. One bwtween "rnn" and "transformer"')

parser.add_argument('--seed', type=str, default='This is my greatest treasure said the small bald creature',
                    help='Initial text of the chapter')

parser.add_argument('--length', type=int, default=25,
                    help='Lenght of generated text')

parser.add_argument('--stochastic',         action='store_true',
                    help='If to generate words sampling from the distribution instead of getting the max')

parser.add_argument('--model_dir',    type=str, default='model/params/',
                    help='Where to save models and params')



def generate_word(net_out, encoder, stochastic, tau=0.1):
    # Initialize softmax
    softmax = nn.Softmax(dim=1)
    # Compute probabilities
    prob = softmax(net_out[:, -1, :]/tau)
    if stochastic:
        # probs should be of size batch x classes
        prob_dist = torch.distributions.Categorical(prob)
        index = prob_dist.sample()
    else:
        # Pick most probable index
        index = torch.argmax(prob).item()
    # Retrieve corresponding word
    text = encoder.decode_text([index])
    return text



def load_RNN(args):
    ### Load training parameters
    model_dir = Path(args.model_dir)
    print ('Loading model from: %s' % model_dir)
    training_args = json.load(open(model_dir / 'training_args.json'))

    ### Load embedding
    # Set paths
    new_emb_path = 'model/glove/glove_lotr.csv'
    emb = pd.read_csv(new_emb_path, sep=' ', quotechar=None, quoting=3, header=None)
    print("Existing embedding matrix loaded\n")
    # Set the first column as index
    emb.index = emb.iloc[:, 0]
    emb.drop(columns=emb.columns[0], inplace=True)

    ### Load encoder
    word_to_index = json.load(open(model_dir / 'word_to_index.json'))
    encoder = Encoder(word_to_index)

    ### Initialize network
    net = Network(embedding_matrix=emb.to_numpy(),
                  hidden_units=training_args['hidden_units'],
                  layers_num=training_args['layers_num'],
                  dropout_prob=training_args['dropout_prob'],
                  device=device) # FIX

    ### Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))

    return net, encoder



if __name__ == "__main__":

    t = time()
    torch.manual_seed(42)

    ### Parse input arguments
    args = parser.parse_args()
    if args.type not in ["rnn", "trasformer"]:
        raise ValueError("Wrong network type.")

    ### Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    # Load network
    if args.type=='rnn':
        net, encoder = load_RNN(args)
    # else:
        # TODO

    net.to(device)
    net.eval() # Evaluation mode (e.g. disable dropout)

    ### Get the encoded representation of the seed
    # Print initial seed
    seed = args.seed
    print(seed, end=' ', flush=True)

    with torch.no_grad():

        ## Find initial state of the RNN
        # Transform words in the corresponding indices
        seed_encoded = torch.Tensor(encoder.encode_text(seed.lower()))
        # Reshape: batch-like shape
        seed_encoded = torch.reshape(seed_encoded, (1, -1))
        # Move to the selected device
        seed_encoded = seed_encoded.to(device)
        # Forward step
        net_out, net_state = net(seed_encoded)
        # Generate next word
        next_word = generate_word(net_out, encoder, args.stochastic)
        # Add to seed
        seed += ' ' + next_word
        # Init first word
        current_word = next_word
        ## Generate words
        for i in range(args.length):
            # Transform words in the corresponding indices
            seed_encoded = torch.Tensor(encoder.encode_text(seed.lower()))
            # Reshape: batch-like shape
            seed_encoded = torch.reshape(seed_encoded, (1, -1))
            # Move to the selected device
            seed_encoded = seed_encoded.to(device)
            # Forward step
            net_out, net_state = net(seed_encoded, net_state)
            # Generate next word
            next_word = generate_word(net_out, encoder, args.stochastic)
            # Add to seed
            seed += ' ' + next_word
            # Print the current result (little tweak to avoid spaces before punctuation)
            if next_word in [".", ":", ",", ";", "!", "?", "'"]:
                print(current_word, end='', flush=True)
            else:
                print(current_word, end=' ', flush=True)
            current_word = next_word

    print('\n', flush=True)
