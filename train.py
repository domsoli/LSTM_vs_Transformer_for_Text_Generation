
# -*- coding: utf-8 -*-

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from dataset import *
from network import Network

# Parameters
parser = argparse.ArgumentParser(description='Train the text generator network.')

# Dataset
parser.add_argument('--datasetpath',    type=str,   default='lotr.txt',
                    help='Path of the train txt file')
parser.add_argument('--crop_len',       type=int,   default=10,
                    help='Number of input words')

# Network
parser.add_argument('--hidden_units',   type=int,   default=2**9,
                    help='Number of RNN hidden units')
parser.add_argument('--layers_num',     type=int,   default=2,
                    help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.3,
                    help='Dropout probability')

# Optimizer
parser.add_argument('--optimizer_lr',   type=float,   default=1e-3,
                    help='Optimizer learning rate')
parser.add_argument('--optimizer_wd',   type=float,   default=1e-4,
                    help='Optimizer weight decay')

# Training
parser.add_argument('--batchsize',      type=int,   default=50,
                    help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=1000,
                    help='Number of training epochs')

# Save
parser.add_argument('--reload',         action='store_true',
                    help='If to reload a previous model')
parser.add_argument('--out_dir',        type=str,   default='model/params/',
                    help='Where to save models and params')



def generate_text(n, state, words, net, w2i, ntokens, device = 'cuda'):

    # Extract last word
    word = state.split()[-1]
    # Handle the situation where the seed is not contained in the dictionary
    if word in words:
        input = torch.tensor(np.reshape(w2i(word), (1, -1))).long().to(device)
    else:
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    # Generate next word
    with torch.no_grad():  # no tracking history
        for i in range(n):
            # Get output
            output = net(input, False)
            word_weights = output[-1].squeeze().exp().cpu()

            # Sample word from output distribution
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)

            # Concatenate the word predicted with the current state
            input = torch.cat([input, word_tensor], 0)
            word = w2i.decoder[word_idx.item()]
            state = '{} {}'.format(state, word)

    # Set punctuations signs and upper case signs
    punc = ['!', '?', '.', ';', ':', ',',"'"]
    upcase = ['?',  '!',  '.']

    # Set initial params
    after_point = False
    new_line_counter = 0
    previous = '_'

    # Print initial state
    print('TEXT:')
    print('{}'.format(state.split()[0]), end = '')

    # Print next word following some given rules
    for i in state.split()[1:]:
        # Avoid loops
        if i == previous:
            continue

        # Update
        previous = i

        # Increment
        new_line_counter += 1

        # Flag: next word capitalized
        if i in upcase:
          after_point = True

        # Flag: start newline after a full point
        if i == '.' and new_line_counter > 10:
          new_line_counter = 0
          print('.')

        # Flag: do not add whitespace, there is punctuation
        elif i in punc:
          print(i, end='')
          new_line_counter -= 1

        # Print new word following flags
        else:
          if after_point:
            if new_line_counter > 1:
                print(' {}'.format(i.capitalize()), end='')
                after_point=False
            # After newline, no whitespace added
            else:
                print('{}'.format(i.capitalize()), end='')
                after_point=False
          else:
            print(' {}'.format(i), end='')



if __name__ == "__main__":

    torch.manual_seed(42)

    # Parse input arguments
    args = parser.parse_args()

    # Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    # Set paths
    json_path = 'model/char_cleaning.json'
    emb_path = 'model/glove/glove.6B.50d.txt'
    new_emb_path = 'model/glove/glove_lotr.csv'

    # Load and clean text
    text = clean_text(args.datasetpath, json_path)

    # Load embedding
    if os.path.exists(new_emb_path):
        emb = pd.read_csv(new_emb_path, sep=' ', quotechar=None, quoting=3, header=None)
        print("Existing embedding matrix loaded")
        # Set the first column as index
        emb.index = emb.iloc[:, 0]
        emb.drop(columns=emb.columns[0], inplace=True)
    else:
        print("Creating new embedding matrix...")
        emb = embedding_matrix(in_path=emb_path,
                               text=text,
                               min_occ=0,
                               out_path=new_emb_path,
                               verbose=False)
        print("Embedding matrix shape:", emb.shape)

    # Define dataset
    trans = transforms.Compose([RandomCrop(args.crop_len),
                                ToTensor()])
    dataset = BookDataset(text, emb, min_len=25, transform=trans)
    print("Dataset lenght: ", len(dataset))

    # Split in train and validation
    train_len = int(len(dataset)*0.8)
    train_set, val_set = random_split(dataset, [train_len, len(dataset)-train_len])

    # Define train and validation dataloader
    train_loader = DataLoader(train_set,
                              batch_size=args.batchsize,
                              pin_memory=True,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=args.batchsize,
                            pin_memory=True,
                            shuffle=False)

    # Initialize Network
    net = Network(embedding_matrix=emb.to_numpy(),
                  hidden_units=args.hidden_units,
                  layers_num=args.layers_num,
                  dropout_prob=args.dropout_prob,
                  device=device) # FIX

    # Move Network into GPU
    net.to(device)

    # Define optimizer
    optimizer = optim.Adam(net.parameters(),
                           lr=args.optimizer_lr,
                           weight_decay=args.optimizer_wd)


    loss_fn = nn.CrossEntropyLoss()

    ### Load the model or save all needed parameters
    if args.reload:
        # Initialize the network with the stored parameters
        net = Network(embedding_matrix=emb.to_numpy(),
                      hidden_units=args['hidden_units'],
                      layers_num=args['layers_num'],
                      dropout_prob=args['dropout_prob'],
                      device=device) # FIX
        ### Load network trained parameters
        net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))
        net.to(device)
        # Load network state
        out_dir = Path(args.out_dir)
        net.load_state_dict(torch.load(out_dir / 'net_params.pth'))
    else:
        # Create output dir
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save training parameters
        with open(out_dir / 'training_args.json', 'w') as f:
            json.dump(vars(args), f, indent=4)
        # Create file to store loss history
        with open(out_dir / 'loss_history.txt', 'w') as f:
            f.write("epoch\ttrain_loss\tval_loss\n")

    # Start training
    for epoch in range(args.num_epochs):
        print('## EPOCH %d' % (epoch + 1))

        # Store batch training loss history
        train_loss = []
        # Train epoch
        for batch_sample in train_loader:
            # Extract batch
            batch_embedded = batch_sample['encoded'].to(device)
            # Update network
            batch_loss, _, _ = net.train_batch(batch_embedded, loss_fn, optimizer)
            train_loss.append(batch_loss)
        train_loss = np.mean(train_loss)

        # Store batch validation loss history
        val_loss = []
        # Validation epoch
        for batch_sample in val_loader:
            # Extract batch
            batch_embedded = batch_sample['encoded'].to(device)
            # Compute validation loss
            batch_loss = net.test_batch(batch_embedded, loss_fn)
            val_loss.append(batch_loss)
        val_loss = np.mean(val_loss)

        print(f'Training loss: {train_loss}\t Validation loss: {val_loss}\n')
        # Create loss history path
        with open(out_dir / 'loss_history.txt', 'a') as f:
            f.write(str(epoch)+"\t"+str(train_loss)+"\t"+str(val_loss)+"\n")

        if epoch % 100 == 0:
            # Print an example of generated text
            # TODO

            # Save network parameters every 100 epoch
            torch.save(net.state_dict(), out_dir / 'net_params.pth')

            # Save encoder dictionary
            with open(out_dir / 'word_to_index.json', 'w') as f:
                json.dump(dataset.word_to_index, f, indent=4)
