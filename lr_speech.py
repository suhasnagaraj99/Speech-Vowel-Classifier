import random
import math
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile
import glob
from numpy import zeros, sign
from math import exp, log
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import argparse

torch.manual_seed(1701)

class SpeechDataset(Dataset):
    def __init__(self, data):
        self.n_samples, self.n_features = data.shape
        self.n_features -= 1
        self.feature = torch.from_numpy(data[:, 1:].astype(np.float32)) 
        self.label = torch.from_numpy(data[:, [0]].astype(np.float32)) 

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.n_samples

def list_files(directory, vowels):
    '''
    Takes in a directory location of the Hillenbrand data and a list of vowels;
    returns a dictionary mapping from vowels to their soundfiles
    '''
    soundfile_dict = {};
    for vowel in vowels:
        soundfile_dict[vowel] = glob.glob(directory+'/*/*'+vowel+'.wav')

    return soundfile_dict

def create_dataset(soundfile_dict, vowels, num_mfccs):
    """
    Read in wav files, and return a 2-D numpy array that contains your
    speech dataset.

    :param soundfile_dict: A dictionary that, for each vowel V, contains a list of file

    paths corresponding to recordings of the utterance 'hVd'

    :param vowels: The set of vowels to be used in the logistic regression

    :param num_mfccs: The number of MFCCs to include as features

    """

    dataset = zeros((len(soundfile_dict[vowels[0]])+len(soundfile_dict[vowels[1]]),num_mfccs+1))

    index = 0
    
    for i,vowel in enumerate(vowels):
        
        for filename in soundfile_dict[vowel]:
            
            utterance, _ = librosa.load(filename,sr=16000)
            
            mfccs = librosa.feature.mfcc(y=utterance, sr=16000, n_mfcc=num_mfccs, n_fft=512, win_length=400, hop_length=160)
            
            if mfccs.shape[1] % 2 == 0:
                
                mid_frame_index = int(mfccs.shape[1]/2) 
                
            else:
                
                mid_frame_index = int((mfccs.shape[1]+1)/2) - 1
                
            midpoint_frame_mfcc = mfccs[:,mid_frame_index]
            
            midpoint_frame_mfcc = np.insert(midpoint_frame_mfcc,0,i)
            
            midpoint_frame_mfcc = midpoint_frame_mfcc.reshape(1,-1)
            
            dataset[index,:] = midpoint_frame_mfcc
            
            index += 1
            
    dataset = np.array(dataset)
    
    mean = np.mean(dataset[:,1:], axis=0)
    
    sd = np.std(dataset[:,1:], axis=0)
    
    dataset[:,1:] = (dataset[:,1:] - mean) / sd

    return dataset


class SimpleLogreg(nn.Module):
    def __init__(self, num_features):
        """
        Initialize the parameters you'll need for the model.

        :param num_features: The number of features in the linear model
        """
        super(SimpleLogreg, self).__init__()
        
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        Compute the model prediction for an example.

        :param x: Example to evaluate
        """
        prediction = torch.sigmoid(self.linear(x))
        return prediction

    def evaluate(self, data):
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc


def step(epoch, ex, model, optimizer, criterion, inputs, labels):
    """Take a single step of the optimizer, we factored it into a single
    function so we could write tests.

    :param epoch: The current epoch
    :param ex: Which example / minibatch you're one
    :param model: The model you're optimizing
    :param inputs: The current set of inputs
    :param labels: The labels for those inputs
    """
    
    model.train()
    
    optimizer.zero_grad()
    
    prediction = model(inputs)
    
    loss = criterion(prediction, labels)
    
    loss.backward()
    
    optimizer.step()

    if (ex+1) % 20 == 0:
      acc_train = model.evaluate(train)
      acc_test = model.evaluate(test)
      print(f'Epoch: {epoch+1}/{num_epochs}, Example {ex}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f} test_acc = {acc_test.item():.4f}')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--vowels", help="The two vowels to be classified, separated by a comma",
                           type=str, default="ih,eh")
    argparser.add_argument("--directory", help="Main directory for the speech files",
                           type=str, default="./Hillenbrand")
    argparser.add_argument("--num_mfccs", help="Number of MFCCs to use",
                           type=int, default=13)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=5)
    argparser.add_argument("--batch", help="Number of items in each batch",
                           type=int, default=1)
    argparser.add_argument("--learnrate", help="Learning rate for SGD",
                           type=float, default=0.1)

    args = argparser.parse_args()

    directory = args.directory
    num_mfccs = args.num_mfccs
    vowels = args.vowels.split(',')

    files = list_files(directory, vowels)
    speechdata = create_dataset(files, vowels, num_mfccs)

    train_np, test_np = train_test_split(speechdata, test_size=0.15, random_state=1234)
    train, test = SpeechDataset(train_np), SpeechDataset(test_np)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    logreg = SimpleLogreg(train.n_features)

    num_epochs = args.passes
    batch = args.batch
    total_samples = len(train)

    criterion = nn.BCELoss()
    
    optimizer = torch.optim.SGD(logreg.parameters(), lr = args.learnrate)

    train_loader = DataLoader(dataset=train,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)
    dataiter = iter(train_loader)

    for epoch in range(num_epochs):
      for ex, (inputs, labels) in enumerate(train_loader):
        step(epoch, ex, logreg, optimizer, criterion, inputs, labels)
