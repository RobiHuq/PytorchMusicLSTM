import Training, Model, miditovector
import time
import os
import pylab

if __name__ == "__main__":
    lrs = [.001, .005,.01, .05, .1]
    hiddensize = [256, 320, 516]
    lstmlayers = [1, 2, 3, 4]
    sequenceSizes = [200, 400, 800, 1200]
    losses = []
    labels =[]
    print("Starting Training")
    lr = .01
    size = 10
    layercount = 1
    seqSize = 10
    dirname = "TrainingResults\\" + \
              "l{}, s{}, lr{}, seq{}".format(layercount, size, lr, seqSize)
    try:
        os.mkdir(dirname)
        print(dirname + "Created")
    except FileExistsError:
        print(dirname + " already exists")
    print("Starting, LR: {}  HS:{} LSTMcount:{}".format(lr, size, layercount))
    rnn = Model.RNN(size, layercount, 1)
    Training.init(rnn, "MusicTensors.pickle", seqSize , lr, ngpu=1)
    startTime = time.time()
    loss = Training.train_set(1000, 5, 10, 10, 50, 50, "Model_lr{}_hs{}_ls{}__{}___".format(lr, size, layercount, seqSize), dir=dirname)
    print("RunTime: " + str(time.time()- startTime))
    losses.append(loss)
    labels.append("l{}, s{}, lr{}, seq{}".format(layercount, size, lr, seqSize))
    #  Creates 20 songs of varrying temperatures
    finaldir = "GeneratedMusic\\" + "l{}, s{}, lr{}, seq{}".format(layercount, size, lr, seqSize)
    try:
        os.mkdir(finaldir)
        print(finaldir + "Created")
    except FileExistsError:
        print(finaldir + " already exists")

    for i in range(1, 21):
        Model.generate(
            rnn, "Final_l{}, s{}, lr{}, seq{}___temp{}.mid".format(layercount, size, lr, seqSize,
                                                               i/20), i/20, length=1000, onGPU=True, dir=finaldir)

    for i in range(labels):
        pylab.plot(losses[i], label=labels[i])
        pylab.legend(loc='upper left')
    pylab.show()




