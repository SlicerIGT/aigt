import os
import sys
import numpy
import random
import pandas
import argparse
import girder_client
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import sklearn
import sklearn.metrics
import cv2
from matplotlib import pyplot as plt
import CNN_LSTM
from CNNLSTMSequence import CNNSequence, LSTMSequence

FLAGS = None

class Train_CNN_LSTM:

    #Loads the data from the specified CSV file
    # fold: The fold number given in the CSV file (should be an int)
    # set: which set the images and labels make up (should be one of: "Train","Validation", or "Test")
    # Returns:
    #   entries.index = A list of index values that define which rows of the datacsv correspond to that fold and set
    # For this network the actual loading of data is handled by the CNNLSTMSequence generator
    def loadData(self,fold,set):
        entries = self.dataCSVFile.loc[(self.dataCSVFile["Fold"] == fold) & (self.dataCSVFile["Set"] == set)]
        return entries.index

    def convertTextToNumericLabels(self,textLabels,labelValues):
        numericLabels =[]
        for i in range(len(textLabels)):
            label = numpy.zeros(len(labelValues))
            labelIndex = numpy.where(labelValues == textLabels[i])
            label[labelIndex] = 1
            numericLabels.append(label)
        return numpy.array(numericLabels)

    def saveTrainingInfo(self,foldNum,saveLocation,trainingHistory,results,networkType):
        LinesToWrite = []
        folds = "Fold " + str(foldNum) +"/"+ str(self.numFolds)
        modelType = "\nNetwork type: " + str(self.networkType)
        LinesToWrite.append(modelType)
        datacsv = "\nData CSV: " + str(FLAGS.data_csv_file)
        LinesToWrite.append(datacsv)
        if networkType == "CNN":
            numEpochs = "\nNumber of Epochs: " + str(self.numEpochs)
            numEpochsInt = self.numEpochs
        else:
            numEpochs = "\nNumber of Epochs: " + str(self.numLSTMEpochs)
            numEpochsInt = self.numLSTMEpochs
        LinesToWrite.append(numEpochs)
        batch_size = "\nBatch size: " + str(self.batch_size)
        LinesToWrite.append(batch_size)
        if networkType == "LSTM":
            lstmSequenceLength = "\nSequence Length: " + str(self.sequenceLength)
            LinesToWrite.append(lstmSequenceLength)
            lstmSamplingRate = "\nDown sampling rate: " + str(self.downsampling)
            LinesToWrite.append(lstmSamplingRate)
            LearningRate = "\nLearning rate: " + str(self.lstm_learning_rate)
        else:
            LearningRate = "\nLearning rate: " + str(self.cnn_learning_rate)
        LinesToWrite.append(LearningRate)
        LossFunction = "\nLoss function: " + str(self.loss_Function)
        LinesToWrite.append(LossFunction)
        trainStatsHeader = "\n\nTraining Statistics: "
        LinesToWrite.append(trainStatsHeader)
        trainLoss = "\n\tFinal training loss: " + str(trainingHistory["loss"][numEpochsInt-1])
        LinesToWrite.append(trainLoss)
        for i in range(len(self.metrics)):
            trainMetrics = "\n\tFinal training " + self.metrics[i] + ": " + str(trainingHistory[self.metrics[i]][numEpochsInt-1])
            LinesToWrite.append(trainMetrics)
        valLoss = "\n\tFinal validation loss: " + str(trainingHistory["val_loss"][numEpochsInt - 1])
        LinesToWrite.append(valLoss)
        for i in range(len(self.metrics)):
            valMetrics = "\n\tFinal validation " + self.metrics[i] + ": " + str(trainingHistory["val_"+self.metrics[i]][numEpochsInt-1])
            LinesToWrite.append(valMetrics)
        testStatsHeader = "\n\nTesting Statistics: "
        LinesToWrite.append(testStatsHeader)
        testLoss = "\n\tTest loss: " + str(results[0])
        LinesToWrite.append(testLoss)
        for i in range(len(self.metrics)):
            testMetrics = "\n\tTest " + self.metrics[i] + ": " + str(results[i+1])
            LinesToWrite.append(testMetrics)

        with open(os.path.join(saveLocation,"trainingInfo_"+networkType+".txt"),'w') as f:
            f.writelines(LinesToWrite)

    def saveTrainingPlot(self,saveLocation,history,metric,networkType):
        fig = plt.figure()
        if networkType == "CNN":
            numEpochs = self.numEpochs
        else:
            numEpochs = self.numLSTMEpochs
        plt.plot([x for x in range(numEpochs)], history[metric], 'bo', label='Training '+metric)
        plt.plot([x for x in range(numEpochs)], history["val_" + metric], 'b', label='Validation '+metric)
        plt.title(networkType+' Training and Validation ' + metric)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(saveLocation, networkType+'_'+metric + '.png'))

    def splitVideoIntoSequences(self, images,sequenceLength=5, downsampling=2):
        number_of_sequences = len(images) - ((sequenceLength) * downsampling)
        sequences = []
        for x in range(0, number_of_sequences):
            sequences.append([i for i in range(x, (x + sequenceLength * downsampling), downsampling)])
        return sequences

    def train(self):
        self.saveLocation = FLAGS.save_location
        self.networkType = os.path.basename(os.path.dirname(self.saveLocation))
        self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
        self.numEpochs = FLAGS.num_epochs_cnn
        self.numLSTMEpochs = FLAGS.num_epochs_lstm
        self.batch_size = FLAGS.batch_size
        self.sequenceLength = FLAGS.sequence_length
        self.downsampling = FLAGS.downsampling_rate
        self.cnn_learning_rate = FLAGS.cnn_learning_rate
        self.lstm_learning_rate = FLAGS.lstm_learning_rate
        self.cnn_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.cnn_learning_rate)
        self.lstm_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.lstm_learning_rate)
        self.loss_Function = FLAGS.loss_function
        self.metrics = FLAGS.metrics.split(",")
        self.numFolds = self.dataCSVFile["Fold"].max() + 1
        self.gClient = None
        network = CNN_LSTM.CNN_LSTM()
        for fold in range(0,self.numFolds):
            foldDir = self.saveLocation+"_Fold_"+str(fold)
            os.mkdir(foldDir)
            taskLabelName = "Task" #This should be the label that will be used to train the network
            toolLabelName = "Tool"

            cnnTrainIndexes = self.loadData(fold,"Train")
            cnnValIndexes = self.loadData(fold, "Validation")
            cnnTestIndexes = self.loadData(fold, "Test")

            if "GirderID" in self.dataCSVFile.columns:
                self.gClient = girder_client.GirderClient(apiUrl=self.dataCSVFile["Girder_URL"][0])
                print("Accessing girder server, authentication required")
                userName = input("Username: ")
                password = input("Password: ")
                self.gClient.authenticate(username=userName, password=password)
                username = os.environ['username']
                tempFileDir = os.path.join('C:/Users/', username, 'Documents/temp')
            else:
                self.gClient = None
                tempFileDir = None
            cnnTrainDataSet = CNNSequence(self.dataCSVFile,cnnTrainIndexes,self.batch_size,toolLabelName,self.gClient,tempFileDir)
            cnnValDataSet = CNNSequence(self.dataCSVFile, cnnValIndexes, self.batch_size, toolLabelName,self.gClient,tempFileDir)
            cnnTestDataSet = CNNSequence(self.dataCSVFile, cnnTestIndexes, self.batch_size, toolLabelName,self.gClient,tempFileDir)

            cnnLabelValues = self.dataCSVFile[toolLabelName].unique()
            numpy.savetxt(os.path.join(foldDir,"cnn_labels.txt"),cnnLabelValues,fmt='%s',delimiter=',')


            cnnModel = network.createCNNModel((224,224,3),num_classes=len(cnnLabelValues))
            cnnModel.compile(optimizer = self.cnn_optimizer, loss = self.loss_Function, metrics = self.metrics)

            history = cnnModel.fit_generator(generator=cnnTrainDataSet,
                                   validation_data=cnnValDataSet,
                                   epochs=self.numEpochs)

            results = cnnModel.evaluate(x=cnnTestDataSet)
            self.saveTrainingInfo(fold, foldDir, history.history, results,"CNN")
            self.saveTrainingPlot(foldDir, history.history, "loss", "CNN")
            for metric in self.metrics:
                self.saveTrainingPlot(foldDir,history.history,metric,"CNN")

            lstmTrainSequences = self.splitVideoIntoSequences(cnnTrainIndexes,sequenceLength=self.sequenceLength,downsampling=self.downsampling)
            lstmValSequences = self.splitVideoIntoSequences(cnnValIndexes,sequenceLength=self.sequenceLength,downsampling=self.downsampling)
            lstmTestSequences = self.splitVideoIntoSequences(cnnTestIndexes,sequenceLength=self.sequenceLength,downsampling=self.downsampling)

            lstmTrainDataSet = LSTMSequence(self.dataCSVFile, cnnTrainIndexes, lstmTrainSequences, cnnModel, self.batch_size, taskLabelName,tempFileDir)
            print("Training images loaded")
            lstmValDataSet = LSTMSequence(self.dataCSVFile, cnnValIndexes, lstmValSequences, cnnModel, self.batch_size, taskLabelName,tempFileDir)
            print("Validation images loaded")
            lstmTestDataSet = LSTMSequence(self.dataCSVFile, cnnTestIndexes, lstmTestSequences, cnnModel, self.batch_size, taskLabelName,tempFileDir)
            print("Test images loaded")

            lstmLabelValues = self.dataCSVFile[taskLabelName].unique()
            numpy.savetxt(os.path.join(foldDir, "lstm_labels.txt"), lstmLabelValues, fmt='%s', delimiter=',')

            lstmModel = network.createLSTMModel(self.sequenceLength, numClasses=len(lstmLabelValues))
            lstmModel.compile(optimizer=self.lstm_optimizer, loss=self.loss_Function, metrics=self.metrics)
            history = lstmModel.fit_generator(generator=lstmTrainDataSet,
                                   validation_data=lstmValDataSet,
                                   epochs=self.numLSTMEpochs)

            results = lstmModel.evaluate(x=lstmTestDataSet)
            self.saveTrainingInfo(fold, foldDir, history.history, results, "LSTM")
            self.saveTrainingPlot(foldDir, history.history, "loss","LSTM")
            for metric in self.metrics:
                self.saveTrainingPlot(foldDir, history.history, metric, "LSTM")


            network.saveModel(cnnModel,lstmModel,foldDir)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='',
      help='Name of the directory where the models and results will be saved'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='',
      help='Path to the csv file containing locations for all data used in training'
  )
  parser.add_argument(
      '--num_epochs_cnn',
      type=int,
      default=10,
      help='number of epochs used in training'
  )
  parser.add_argument(
      '--num_epochs_lstm',
      type=int,
      default=10,
      help='number of epochs used in training'
  )
  parser.add_argument(
      '--sequence_length',
      type=int,
      default=50,
      help='number of epochs used in training'
  )
  parser.add_argument(
      '--downsampling_rate',
      type=int,
      default=4,
      help='number of epochs used in training'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=64,
      help='type of output your model generates'
  )
  parser.add_argument(
      '--cnn_learning_rate',
      type=float,
      default=0.0001,
      help='Learning rate used in training cnn network'
  )
  parser.add_argument(
      '--lstm_learning_rate',
      type=float,
      default=0.00001,
      help='Learning rate used in training lstm network'
  )
  parser.add_argument(
      '--loss_function',
      type=str,
      default='categorical_crossentropy',
      help='Name of the loss function to be used in training (see keras documentation).'
  )
  parser.add_argument(
      '--metrics',
      type=str,
      default='accuracy',
      help='Metrics used to evaluate model.'
  )
FLAGS, unparsed = parser.parse_known_args()
tm = Train_CNN_LSTM()
tm.train()
