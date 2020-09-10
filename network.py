# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self,
                embedding_matrix,
                hidden_units,
                layers_num,
                device,
                dropout_prob=0,
                train_emb=False
                ):

        # Call the parent init function
        super(Network, self).__init__()

        # Set dimensions
        embedding_tensor = torch.from_numpy(embedding_matrix).to(device).float()
        vocab_size = embedding_tensor.size()[0]
        embedding_dim = embedding_tensor.size()[1]

        # Define embedding layer (load pretrained)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_tensor,
            freeze=not train_emb
        )

        # Define recurrent layer
        self.rnn = nn.LSTM(
            # Define size of the one-hot-encoded input
            input_size=embedding_dim,
            # Define size of a single recurrent hidden layer
            hidden_size=hidden_units,
            # Define number of stacked recurrent hidden layers
            num_layers=layers_num,
            # Set dropout probability
            dropout=dropout_prob,
            # Set batch size as first dimension
            batch_first=True)

        # Define linear layer
        self.out = nn.Linear(hidden_units, vocab_size)

        # Save parameters in the model
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.layers_num = layers_num


    def forward(self, x, state=None):
        # Embedding layer
        x = self.embedding(x.long())
        # LSTM
        x, rnn_state = self.rnn(x)
        # Linear
        x = self.out(x)
        # # Softmax
        # softmax = nn.Softmax(dim=0)
        # x = softmax(x)

        return x, rnn_state


    def train_batch(self, batch, loss_fn, optimizer):

        # Get the labels (the last word of each sequence)
        labels = batch[:, -1]
        # Remove the labels from the input tensor
        input = batch[:, :-1]
        # Eventually clear previous recorded gradients
        optimizer.zero_grad()
        # Forward pass
        output, _ = self(input)
        # Evaluate loss only for last output
        loss = loss_fn(output[:, -1, :].float(), labels.long())
        # Backward pass
        loss.backward()
        # Update network
        optimizer.step()

        # Return average batch loss
        return float(loss.data), output, labels


    def test_batch(self, batch, loss_fn):

        # Get the labels (the last word of each sequence)
        labels = batch[:, -1]
        # Remove the labels from the input tensor
        input = batch[:, :-1]
        # Make forward pass
        output, _ = self(input)
        # Evaluate loss (only for last output)
        loss = loss_fn(output[:, -1, :].float(), labels.long())

        # Return average batch loss
        return float(loss.data)



if __name__ == "__main__":

    import time
    from dataset import *

    torch.manual_seed(42)

    # Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    # Set arguments
    crop_len = 10
    batchnumber = 100
    hidden_units = 128
    layers_num = 2
    dropout_prob = 0.3

    # Set paths
    text_path = 'lotr.txt'
    json_path = 'model/char_cleaning.json'
    emb_path = 'model/glove/glove.6B.50d.txt'
    new_emb_path = 'model/glove/glove_lotr.csv'

    # Load and clean text
    text = clean_text(text_path, json_path)

    # Load embedding
    if os.path.exists(new_emb_path):
        emb = pd.read_csv(new_emb_path, sep=' ', quotechar=None, quoting=3, header=None)
        # Set the first column as index
        emb.index = emb.iloc[:, 0]
        emb.drop(columns=emb.columns[0], inplace=True)
    else:
        emb = embedding_matrix(emb_path, text, min_occ=0, out_path=new_emb_path, verbose=True)

    # Define dataset and dataloader
    trans = transforms.Compose([RandomCrop(crop_len),
                                ToTensor()])
    dataset = BookDataset(text, emb, min_len=15, transform=trans)
    dataloader = DataLoader(dataset, batch_size=100)

    # Initialize Network
    net = Network(embedding_matrix=emb.to_numpy(),
                  hidden_units=hidden_units,
                  layers_num=layers_num,
                  dropout_prob=dropout_prob,
                  device=device)

    # Move Network into GPU
    net = net.to(device)
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.003)
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    ### Init Training
    train_loss = []
    # Start training
    print('Start EPOCH')

    b_losses = []
    # Iterate batches
    for batch_sample in dataloader:
        # Extract batch
        batch_onehot = batch_sample['encoded'].to(device)
        print("Batch shape:", list(batch_onehot.shape))
        # Update network
        batch_loss, out, y_true = net.train_batch(batch_onehot, loss_fn, optimizer)
        b_losses.append(batch_loss)
        optimizer.zero_grad()
