import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import miditovector


class RNN(nn.Module):
    def __init__(self, hiddenSize, lstmlayers, ngpu=0):
        super(RNN, self).__init__()
        self.npu = ngpu
        self.hiddenSize = hiddenSize
        self.lstmlayers = lstmlayers
        self.encoder = nn.Embedding(129, hiddenSize)
        self.lstm = nn.LSTM(hiddenSize, hiddenSize, lstmlayers)
        self.dense = nn.Linear(hiddenSize, 129)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.lstmlayers, 1, self.hiddenSize)),
                Variable(torch.zeros(self.lstmlayers, 1, self.hiddenSize)))  # one is for batch siz

    def forward(self, x, hidden):
        x = self.encoder(x)
        x = F.relu(x)
        x, hidden = self.lstm(x.view(1, 1, -1), hidden)
        x = self.dense(x.view(1, -1))
        return x, hidden


def generate(model, songname, temperature, primer=[-1], length=500, dir="GeneratedMusic", onGPU=False):

    # Randomize primer if none set
    p = [-1]
    if p == primer:
        primer = [random.randint(0, 128)]

    with torch.no_grad():
        model.training = False
        hidden = model.init_hidden()
        primeTensor = torch.tensor(primer).long()
        if onGPU:
            primeTensor.cuda()

        # prime the hidden
        for p in range(len(primeTensor)):
            _, hidden = model(primeTensor[p], hidden)
        inp = primeTensor[-1]  # grab the end of the prime tensor to use for the creation
        song = []
        for note in range(length):
            output, hidden = model(inp, hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            song.append(top_i)
            inp = top_i
        miditovector.noteIndexesToSong(song, songname, dir=dir)
        model.training = True
