from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from features import mfcc, logfbank
from scipy.io import wavfile
from pybrain.datasets import SupervisedDataSet as SDS
from pybrain.supervised.trainers import BackpropTrainer

#reads wav file and transforms it
datatset = SDS(13, 1)
for i in range(10)
	rate, data = wavfile.read("audio%d.wav" % i)
	mfcc_feat = mfcc(data,rate)
	datachunk = mfcc_feat.mean(axis=0) #averages all of the feature vectors
	if i % 2 == 0:
		target = "0"
	else:
		target = "1"
	dataset.addSample(datachunk, target);

#sets up data

#creates the FFN 
net = FeedForwardNetwork()

#creates 13 input perceptrons, 50 hidden perceptrons, and 1 output perceptron
inputLayer = LinearLayer(13)
hiddenLayer = SigmoidLayer(50)
outputLayer = LinearLayer(1)

#adds them to the FFN
net.addInputModule(inputLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outputLayer)

#connects all of the perceptrons between two layers together
inputConnect = FullConnection(inputLayer, hiddenLayer)
outputConnect = FullConnection(hiddenLayer, outputLayer)

#adds these connections to the FFN
net.addConnection(inputConnect)
net.addConnection(outputConnect)

net.sortModules()

#sets up the trainer
trainer = BackpropTrainer(net, dataset)
#taken from pybrain tutorial to train and show errors
for i in range (20):
	trnresult = percentError( trainer.testOnClassData(), 
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult