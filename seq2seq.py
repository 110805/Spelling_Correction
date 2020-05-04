# -*- coding: utf-8 -*-
"""seq2seq.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/110805/Spelling_Correction/blob/master/seq2seq.ipynb
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import json



"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
2. Output your results (BLEU-4 score, correction words)
3. Plot loss/score
4. Load/save weights
========================================================================================"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
vocab_size = 28
teacher_forcing_ratio = 0.7
LR = 0.01
MAX_LENGTH = 20

################################
#Example inputs of compute_bleu
################################
#The target word
reference = 'variable'
#The word generated by your model
output = 'varable'

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def sample_pair(i, Data):
    input_tensor = []
    target_tensor = []
    
    for input_char in Data[i][0]:
        input_tensor.append(ord(input_char)-95)
    
    for target_char in Data[i][1]:
        target_tensor.append(ord(target_char)-95)

    target_tensor.append(EOS_token)
    return (torch.tensor(input_tensor, dtype=torch.long).view(-1, 1), torch.tensor(target_tensor, dtype=torch.long).view(-1, 1))

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    #----------sequence to sequence part for encoder----------#
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	
    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(encoder, decoder, input_string, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = []
        for input_char in input_string:
            input_tensor.append(ord(input_char)-95)

        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1, 1)
    
        input_tensor = input_tensor.to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(chr(topi.item()+95))

            decoder_input = topi.squeeze().detach()

        pred = ''
        for i in range(len(decoded_words)):
            pred += decoded_words[i]

        return pred

def evalTestdata(encoder, decoder):
    score = 0
    with open('test.json') as f:
        voc = json.load(f)
    
    for data in voc:
        output = evaluate(encoder, decoder, data['input'][0])
        #print('input: {}'.format(data['input'][0]))
        #print('target: {}'.format(data['target']))
        #print('pred: {}'.format(output))
        
        if len(output) != 0:
            score += compute_bleu(output, data['target'])
        else:
            score += compute_bleu('', data['target']) # predict empty string
        
        #print('--------------------')
    #print('BLEU-4 score:{}'.format(score/50))
    return score/50
    
def trainIters(encoder, decoder, n_epochs, learning_rate=LR):
    start = time.time()
    plot_losses = []
    BLEU_scores = []
    epoch_loss = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    Data = []
    with open('train.json') as f:
        Voc = json.load(f)

    for voc in Voc:
        for i in range(len(voc['input'])):
            group = []
            group.append(voc['input'][i])
            group.append(voc['target'])
            Data.append(group)

    training_pairs = [sample_pair(i, Data) for i in range(len(Data))]
    print('Finish sampling')
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, n_epochs + 1):
        for iter in range(len(Data)):
            training_pair = training_pairs[iter]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
        
            epoch_loss += loss

        epoch_loss_avg = epoch_loss / len(Data) 
        plot_losses.append(epoch_loss_avg)
        epoch_loss = 0
        bleu_score = evalTestdata(encoder, decoder)
        BLEU_scores.append(bleu_score)
        print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, epoch_loss_avg, bleu_score))

    plt.figure(1)
    plt.plot(range(n_epochs), plot_losses)
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropyLoss')
    plt.savefig('TrainingLoss')

    plt.figure(2)
    plt.plot(range(n_epochs), BLEU_scores)
    plt.xlabel('Epochs')
    plt.ylabel('BLEU_scores')
    plt.savefig('BLEU_scores')

encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
trainIters(encoder1, decoder1, 100)
torch.save(encoder1.state_dict(), 'encoder.pkl')
torch.save(decoder1.state_dict(), 'decoder.pkl')

encoder = EncoderRNN(vocab_size, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, vocab_size).to(device)
encoder.load_state_dict(torch.load('encoder.pkl'))
decoder.load_state_dict(torch.load('decoder.pkl'))

def eval(encoder, decoder):
    score = 0
    with open('new_test.json') as f:
        voc = json.load(f)
                            
        for data in voc:
            output = evaluate(encoder, decoder, data['input'][0])
            print('input: {}'.format(data['input'][0]))
            print('target: {}'.format(data['target']))
            print('pred: {}'.format(output))
                                                                            
            if len(output) != 0:                                                        
                score += compute_bleu(output, data['target'])
            else:
                score += compute_bleu('', data['target']) # predict empty string
            
            print('--------------------')
        
        print('BLEU-4 score:{}'.format(score/50))

eval(encoder, decoder)
