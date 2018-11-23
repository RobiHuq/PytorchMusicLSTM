import os
import random
import pickle
import torch
import torch.nn as nn
import time

import Model

model = None
data = None
sampleLength = None
device = None
num_gpu = None
criterion = None
optimizer = None


# Under the assumption that we have 1 gpu
def init(model_to_train, trainingFile, length, lr, training_device=None, ngpu=1):
    global model, data, sampleLength, device, num_gpu, criterion, optimizer
    model = model_to_train
    num_gpu = ngpu
    try:
        data = pickle.load(open(trainingFile, "rb"))
    except AttributeError:
        print("ERROR :( Could not open data file")
        exit()
    sampleLength = length

    if training_device is None:
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        if ngpu > 0:
            model.to(device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def randomSample():
    r = random.choice(data)
    startindex = random.randint(0, len(r) - sampleLength-2)
    end = startindex + sampleLength + 1
    return r[startindex:end].to(device)


def rand_training_set():
    sample = randomSample()
    inp = sample[:-1]
    target = sample[1:]

    inp = torch.tensor(inp).long()
    target = target.long()
    if num_gpu > 0:
        target.cuda()
        inp.cuda()
    return inp, target


def train(sequences):
    model.zero_grad()
    loss = 0
    for i in range(sequences):
        hidden = model.init_hidden()
        if num_gpu > 0:
            for h in hidden:
                h.to(device)
        inp, target = rand_training_set()
        for noteIndex in range(sampleLength):
            output, hidden = model(inp[noteIndex], hidden)
            if num_gpu > 0:
                t = torch.Tensor([target[noteIndex]]).long().cuda()
            else:
                t = torch.Tensor([target[noteIndex]]).long()

            loss += criterion(output, t)

    loss.backward()
    optimizer.step()
    return loss / (sampleLength * sequences)


def train_set(epochs, batch_size, print_every, plot_every, synthesize_every, save_every, name, dir="TrainingResults"):
    losses = []
    avg_loss = 0
    startTime = time.time()
    for epoch in range(1, epochs + 1):
        loss = train(batch_size)
        avg_loss += avg_loss
        if epoch % print_every == 0:
            print("TimeperEpoch: {}, Epoch:[{}/{}, Loss: {}".format((time.time() - startTime) / print_every, epoch,
                                                                    epochs, loss))
            startTime = time.time()
        if epoch % plot_every == 0:
            losses.append(avg_loss / plot_every)
            avg_loss = 0
        if epoch % synthesize_every == 0:
            train_output = dir + "\\" + str(epoch)
            try:
                os.mkdir(train_output)
                print(train_output + "Created")
            except FileExistsError:
                print(train_output + " already exists")
            print("Generating Samples")
            for i in range(1, 21):
                Model.generate(model, "{}___temp{}.mid".format(name, i / 20), i / 20, length=1200, onGPU=True, dir=train_output)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), "Models\{}{}_{}.pt".format(name, epoch, time.time()))

    return losses

