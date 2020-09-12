
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
parser.add_argument('--hidden_units',   type=int,   default=2**8,
                    help='Number of RNN hidden units')
parser.add_argument('--layers_num',     type=int,   default=2,
                    help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.2,
                    help='Dropout probability')

# Optimizer
parser.add_argument('--optimizer_lr',   type=float,   default=1e-3,
                    help='Optimizer learning rate')
parser.add_argument('--optimizer_wd',   type=float,   default=0,
                    help='Optimizer weight decay')

# Training
parser.add_argument('--batchsize',      type=int,   default=100,
                    help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=1000,
                    help='Number of training epochs')

# Save
parser.add_argument('--reload',         action='store_true',
                    help='If to reload a previous model')
parser.add_argument('--out_dir',        type=str,   default='model/params/',
                    help='Where to save models and params')



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
