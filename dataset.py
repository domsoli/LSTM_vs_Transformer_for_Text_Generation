
# -*- coding: utf-8 -*-

import os
import re
import json
import torch
import argparse
import numpy as np
import pandas as pd
from torch import optim, nn
from functools import reduce
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class BookDataset(Dataset):

    def __init__(self, text, emb, min_len, transform=None):
        """
        Attributes:
        - text (str) book text
        - min_len (int) minimum acceptable sentence lenght
        - transform (torchvision.transform) trandformation to be applied
        """
        # Call the parent init function
        super(Dataset, self).__init__()

        # Extract the sentences
        sentences = re.split('[.]', text)
        sentences = [i for i in sentences if len(i.split()) > min_len]
        sentence_list = sentences

        ### Words are mapped into embedded matrix indexes
        word_to_index = {key: value for value, key in enumerate(emb.index)}
        index_to_word = {word_to_index[key]: key for key in word_to_index}

        # Store data
        self.corpus = text
        self.vocabolary_len = emb.shape[0]
        self.sentence_list = sentence_list
        self.transform = transform
        self.emb = emb
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word


    def __len__(self):
        return len(self.sentence_list)


    def __getitem__(self, idx):
        # Get text
        text = self.sentence_list[idx]
        # Encode text
        encoded_list = []
        for c in re.findall(r"[']+|[\w']+|[.,!?;:]", text):
            if c in self.word_to_index.keys():
                encoded_list.append(self.word_to_index[c])
        # Create sample
        sample = {'text': text, 'encoded': encoded_list}
        # Transform (if defined)
        if self.transform:
            sample = self.transform(sample)
        return sample



class Encoder(object):

    def __init__(self, word_to_index):
        # Reverse the dictionary
        index_to_word = {word_to_index[key]: key for key in word_to_index}
        # Store the two maps
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def encode_text(self, decoded): # TODO move it in a separate class
        """
        Input:
        - decoded (str) Original text to encode
        """
        word_list = decoded.split()
        encoded_text = [self.word_to_index[c] for c in word_list if c in self.word_to_index]
        return encoded_text

    def decode_text(self, encoded): # TODO move it in a separate class
        """
        Input:
        - encoded (list) List containing the indexes of encoded words
        """
        encoded = [int(n) for n in encoded]
        text = [self.index_to_word[c] for c in encoded if c in self.index_to_word]
        return " ".join(text)



class RandomCrop():

    def __init__(self, crop_len):
        # Lenght of the chuncks of sentences to be sampled
        self.crop_len = crop_len

    def __call__(self, sample):
        text = sample['text']
        encoded = sample['encoded']
        # Randomly choose first and last words
        words_list = re.findall(r"[']+|[\w']+|[.,!?;:]", text)
        start_words = np.random.randint(0, len(words_list) - self.crop_len)
        end_words = start_words + self.crop_len
        # Generate text using the chosen words
        cropped_text = ' '.join(words_list[start_words: end_words])
        # print(len(text.split()))
        if len(cropped_text.split()) < self.crop_len:
            print("WARNING - Irregular lenght")
            print(len(cropped_text.split()))
            print(cropped_text.split())
        return {**sample,
                'text': cropped_text,
                'encoded': encoded[start_words: end_words]}



class ToTensor():

    def __call__(self, sample):
        # Convert encoded text to pytorch tensor
        encoded = torch.tensor(sample['encoded']).float()
        return {'text': sample['text'],
                'encoded': encoded}



def embedding_matrix(in_path, text, out_path):
    """
    ****************
    Arguments:
    - emb      (str)  path to embedding matrix
    - text     (str)  cleaned text
    - min_occ  (int)  minimum number of occurrences to keep a word
    - out_path (str)  path to store the new embedding matrix
    - verbose  (bool) verbose
    ****************
    Output:
    - emb      (pandas.DataFrame)  new embedding matrix
    ****************
    """
    # Import glove data as a DataFrame
    emb = pd.read_csv(in_path, sep=' ', quotechar=None, quoting=3, header=None)
    # Set the first column as index
    emb.index = emb.iloc[:, 0]
    emb.drop(columns=emb.columns[0], inplace=True)
    # Save the words corpus with occurrences
    corpus = set(re.findall(r"[']+|[\w']+|[.,!?;:]", text))
    # Check the words already mapped in the embedding matrix
    word_in_emb = [i for i in corpus if i in emb.index]
    word_out_emb = [i for i in corpus if i not in emb.index]
    # Save the only vectors corresponding to words present in corpus
    emb = emb.loc[word_in_emb, :]
    emb = pd.DataFrame(emb.values, index=emb.index, dtype=float)
    # Add rows for words in text not present in emb
    for word in word_out_emb:
        emb.loc[word, :] = np.random.rand(emb.shape[1])
    # Check that there are no duplicates
    # TODO
    # Normalize the vectors
    emb = emb.apply(lambda x: x/np.linalg.norm(x), axis=1)
    # Save the matrix
    emb.to_csv(out_path, sep = ' ', quotechar=None, quoting=3)
    return emb



def clean_text(path, json_path):
    """
    ****************
    Arguments:
    - path      (str) path of the text file to be cleaned;
    - json_path (str) path of the json containing the char substitutions.
    ****************
    Output:
    - text      (str) cleaned text
    ****************
    """
    # Load text
    with open(path, 'r') as file:
        text = file.read()
    # Load cleaning json
    with open(json_path, 'r') as json_file:
        char_dict = json.load(json_file)
    # Remove special characters
    for c in char_dict:
        text = text.replace(c, char_dict[c])
    # Remove spaces after a new line
    text = re.sub('\n[ ]+', '\n', text)
    # Lower case
    text = text.lower()

    return text



if __name__ == "__main__":

    # Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    # Set arguments
    crop_len = 10
    batchnumber = 100
    reload = False

    # Set paths
    text_path = 'lotr.txt'
    json_path = 'model/char_cleaning.json'
    emb_path = 'model/glove/glove.6B.50d.txt'
    new_emb_path = 'model/glove/glove_lotr.csv'

    # Load text and embeddings
    text = clean_text(text_path, json_path)

    if reload:
        if os.path.exists(new_emb_path):
            emb = pd.read_csv(new_emb_path, sep=' ', quotechar=None, quoting=3, header=None)
            # Set the first column as index
            emb.index = emb.iloc[:, 0]
            emb.drop(columns=emb.columns[0], inplace=True)
        else:
            print("Path not found - Computing new matrix...")
            emb = embedding_matrix(emb_path, text, out_path=new_emb_path)
    else:
        emb = embedding_matrix(emb_path, text, out_path=new_emb_path)

    print("Original embedding matrix shape", emb.shape)
    print(emb.head())

    print("Embedded matrix shape:", emb.shape)

    ## Define dataset
    trans = transforms.Compose([RandomCrop(crop_len),
                                ToTensor()
                                ])

    dataset = BookDataset(text, emb, min_len=25, transform=trans)
    print("Found {} sentences".format(len(dataset.sentence_list)))

    # Define Encoder
    encoder = Encoder(dataset.word_to_index)


    #%% Test sampling
    sample = dataset[0]

    print('\n*** TEXT SAMPLE ***')
    print('Original text:\t\t', sample['text'])
    print('Reconstcucted text:\t', encoder.decode_text(sample['encoded']))
    print('*******************\n')

    cont_irr = 0
    cont_reg = 0
    for i in dataset:
        if (len(i['encoded']) != 10):
            cont_irr += 1
            print('Text lenght is ', len(i['text'].split()))
            print('Original text:\t', i['text'], '\n')
            print('Reconstcucted text:\t', " ".join(encoder.decode_text(i['encoded'])))
            print('\n')
        else:
            cont_reg += 1

    if cont_irr == 0:
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=100)
        for batch in dataloader:
            print(batch['encoded'].shape)
    else:
        print("ERROR - {} irregular lenght sentences found!".format(cont_irr))
