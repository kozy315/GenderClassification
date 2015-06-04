from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from features import mfcc
from scipy.io import wavfile
from pybrain.datasets import SupervisedDataSet as SDS
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator

#reads wav file, transforms it, and turns it into a data set
ds = SDS(13, 1) #creates the dataset

for i in range(10):
	rate, data = wavfile.read("audio%d.wav" % i)
	mfcc_feat = mfcc(data,rate)
	datachunk = mfcc_feat.mean(axis=0) #averages all of the feature vectors
	#gender alternates every other file so this just sets what the target result is.
	if i % 2 == 0:
		target = "0"
	else:
		target = "1"
	ds.addSample(datachunk, target); #adds the mfcc features to the dataset
	
dsTest = SDS(13, 1)

for i in range(2):
	rate, data = wavfile.read("audioTest%d.wav" % i)
	mfcc_feat = mfcc(data,rate)
	datachunk = mfcc_feat.mean(axis=0) #averages all of the feature vectors
	#gender alternates every other file so this just sets what the target result is.
	if i % 2 == 0:
		target = "0"
	else:
		target = "1"
	dsTest.addSample(datachunk, target); #adds the mfcc features to the dataset
	
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
trainer = BackpropTrainer(net, ds)

#simple train
trainer.trainEpochs(50)
trainer.testOnData(dataset=dsTest, verbose=True)