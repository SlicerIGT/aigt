import os
import sys
import numpy
import random
import pandas
import argparse
import girder_client
import tensorflow
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
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
    def loadData(self,fold,set,dataset):
        entries = dataset.loc[(dataset["Fold"] == fold) & (dataset["Set"] == set)]
        return entries.index

    def convertTextToNumericLabels(self,textLabels,labelValues):
        numericLabels =[]
        for i in range(len(textLabels)):
            label = numpy.zeros(len(labelValues))
            labelIndex = numpy.where(labelValues == textLabels[i])
            label[labelIndex] = 1
            numericLabels.append(label)
        return numpy.array(numericLabels)

    def saveTrainingInfo(self,foldNum,saveLocation,trainingHistory,results,networkType,balanced=False):
        LinesToWrite = []
        modelType = "\nNetwork type: " + str(self.networkType)
        LinesToWrite.append(modelType)
        datacsv = "\nData CSV: " + str(FLAGS.data_csv_file)
        LinesToWrite.append(datacsv)
        numEpochs = "\nNumber of Epochs: " + str(len(trainingHistory["loss"]))
        numEpochsInt = len(trainingHistory["loss"])
        LinesToWrite.append(numEpochs)
        batch_size = "\nBatch size: " + str(self.batch_size)
        LinesToWrite.append(batch_size)
        if networkType == "LSTM":
            lstmSequenceLength = "\nSequence Length: " + str(self.sequenceLength)
            LinesToWrite.append(lstmSequenceLength)
            lstmSamplingRate = "\nDown sampling rate: " + str(self.downsampling)
            LinesToWrite.append(lstmSamplingRate)
            LearningRate = "\nLearning rate: " + str(self.lstm_learning_rate)
            classWeights = '\nClass weights: ' + str(self.lstmClassWeights)
        else:
            LearningRate = "\nLearning rate: " + str(self.cnn_learning_rate)
            classWeights = '\nClass weights: ' + str(self.cnnClassWeights)

        LinesToWrite.append(LearningRate)
        LinesToWrite.append(classWeights)
        dataBalance = "\nData balanced: " + str(balanced)
        LinesToWrite.append(dataBalance)
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

        LinesToWrite.append("\n" + str(self.confMat))

        with open(os.path.join(saveLocation,"trainingInfo_"+networkType+".txt"),'w') as f:
            f.writelines(LinesToWrite)

    def saveTrainingPlot(self,saveLocation,history,metric,networkType):
        fig = plt.figure()
        numEpochs =len(history[metric])
        '''if networkType == "CNN":
            numEpochs = self.numEpochs
        else:
            numEpochs = self.numLSTMEpochs'''
        plt.plot([x for x in range(numEpochs)], history[metric], 'bo', label='Training '+metric)
        plt.plot([x for x in range(numEpochs)], history["val_" + metric], 'b', label='Validation '+metric)
        plt.title(networkType+' Training and Validation ' + metric)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(saveLocation, networkType+'_'+metric + '.png'))
        plt.close(fig)

    def splitVideoIntoSequences(self, images,sequenceLength=5, downsampling=2):
        number_of_sequences = len(images) - ((sequenceLength) * downsampling)
        sequences = []
        prevSequences = [[-1 for i in range(sequenceLength)] for j in range(downsampling)]
        for x in range(sequenceLength):
            for y in range(downsampling):
                newSeq = prevSequences[y][1:]
                newSeq.append(images[(x * downsampling) + y])
                prevSequences[y] = newSeq
                sequences.append(prevSequences[y])
        for x in range(0, number_of_sequences):
            sequences.append([images[i] for i in range(x, (x + sequenceLength * downsampling), downsampling)])
        return sequences

    def splitDatasetIntoSequences(self, fold,set,sequenceLength=5, downsampling=2):
        entries = self.dataCSVFile.loc[(self.dataCSVFile["Fold"]==fold)&(self.dataCSVFile["Set"]==set)]
        videoNames = entries["Folder"].unique()
        allSequences = []
        for video in videoNames:
            videoImages = entries.loc[entries["Folder"]==video]
            videoImageIndexes = videoImages.index
            videoSequences = self.splitVideoIntoSequences(videoImageIndexes,sequenceLength,downsampling)
            for sequence in videoSequences:
                allSequences.append(sequence)
        return allSequences

    def balanceDataset(self,dataset):
        videos = dataset["Folder"].unique()
        balancedFold = pandas.DataFrame(columns=dataset.columns)
        for vid in videos:
            images = dataset.loc[dataset["Folder"] == vid]
            labels = sorted(images["Tool"].unique())
            counts = images["Tool"].value_counts()
            print(vid)
            smallestCount = counts[counts.index[-1]]
            print("Smallest label: " + str(counts.index[-1]))
            print("Smallest count: " + str(smallestCount))
            if smallestCount == 0:
                print("Taking second smallest")
                secondSmallest = counts[counts.index[-2]]
                print("Second smallest count: " + str(secondSmallest))
                reducedLabels = [x for x in labels if x != counts.index[-1]]
                print(reducedLabels)
                for label in reducedLabels:
                    toolImages = images.loc[images["Tool"] == label]
                    randomSample = toolImages.sample(n=secondSmallest)
                    balancedFold = balancedFold.append(randomSample, ignore_index=True)
            else:
                for label in labels:
                    toolImages = images.loc[images["Tool"] == label]
                    if label == counts.index[-1]:
                        balancedFold = balancedFold.append(toolImages, ignore_index=True)
                    else:
                        randomSample = toolImages.sample(n=smallestCount)
                        balancedFold = balancedFold.append(randomSample, ignore_index=True)
        print(balancedFold["Tool"].value_counts())
        return balancedFold

    def createBalancedCNNDataset(self,fold):
        newCSV = pandas.DataFrame(columns=self.dataCSVFile.columns)
        trainSet = self.dataCSVFile.loc[(self.dataCSVFile["Fold"] == fold) & (self.dataCSVFile["Set"] == "Train")]
        resampledTrainSet = self.balanceDataset(trainSet)
        sortedTrain = resampledTrainSet.sort_values(by=['FileName'])
        newCSV = newCSV.append(sortedTrain, ignore_index=True)
        valSet = self.dataCSVFile.loc[(self.dataCSVFile["Fold"] == fold) & (self.dataCSVFile["Set"] == "Validation")]
        resampledValSet = self.balanceDataset(valSet)
        sortedVal = resampledValSet.sort_values(by=['FileName'])
        newCSV = newCSV.append(sortedVal, ignore_index=True)
        testSet = self.dataCSVFile.loc[(self.dataCSVFile["Set"] == "Test") & (self.dataCSVFile["Fold"] == fold)]
        newCSV = newCSV.append(testSet, ignore_index=True)
        print("Resampled Train Counts")
        print(resampledTrainSet["Tool"].value_counts())
        print("Resampled Validation Counts")
        print(resampledValSet["Tool"].value_counts())
        print("Test Counts")
        print(testSet["Tool"].value_counts())
        return newCSV

    def getBalancedSequences(self,sequences):
        sequenceLabels = self.getSequenceLabels(sequences)
        tempDataFrame = pandas.DataFrame({"Sequences":sequences,"Labels":sequenceLabels})
        balancedFold = pandas.DataFrame(columns = tempDataFrame.columns)
        counts = tempDataFrame["Labels"].value_counts()
        print("Initial Counts: ")
        print(counts)
        smallestCount = counts[counts.index[-1]]
        print("Smallest label: " + str(counts.index[-1]))
        print("Smallest count: " + str(smallestCount))
        if smallestCount == 0:
            print("Taking second smallest")
            secondSmallest = counts[counts.index[-2]]
            print("Second smallest count: " + str(secondSmallest))
            reducedLabels = [x for x in self.lstmLabelValues if x != counts.index[-1]]
            print(reducedLabels)
            for label in reducedLabels:
                taskSequences = tempDataFrame.loc[tempDataFrame["Labels"] == label]
                randomSample = taskSequences.sample(n=secondSmallest)
                balancedFold = balancedFold.append(randomSample, ignore_index=True)
        else:
            for label in self.lstmLabelValues:
                taskSequences = tempDataFrame.loc[tempDataFrame["Labels"] == label]
                if label == counts.index[-1]:
                    balancedFold = balancedFold.append(taskSequences, ignore_index=True)
                else:
                    randomSample = taskSequences.sample(n=smallestCount)
                    balancedFold = balancedFold.append(randomSample, ignore_index=True)
        balancedSequences = []
        for i in balancedFold.index:
            balancedSequences.append(balancedFold["Sequences"][i])
        print("Resampled Sequence Counts")
        print(balancedFold["Labels"].value_counts())
        return balancedSequences

    def getSequenceLabels(self, sequences):
        sequenceLabels = []
        for sequence in sequences:
            textLabel = self.dataCSVFile["Task"][sequence[len(sequence) - 1]]
            sequenceLabels.append(textLabel)
        return sequenceLabels

    def getClassWeights(self,trainLabels):
        trainLabelCounts = numpy.sum(trainLabels, axis=0)
        numSamples = numpy.sum(trainLabelCounts)
        weights = [max(round(numSamples / (8.0*count),0),1) for count in trainLabelCounts]
        #weights[2] +=50
        #weights[5] +=50
        labels = [i for i in range(trainLabelCounts.shape[0])]
        classWeights = dict(zip(labels, weights))
        print(classWeights)
        return classWeights

    def train(self):
        balanceCNNData = False
        balanceLSTMData = False
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
            balancedDataset = self.createBalancedCNNDataset(fold)            foldDir = self.saveLocation+"_Fold_"+str(fold)
            os.mkdir(foldDir)
            taskLabelName = "Task" #This should be the label that will be used to train the network
            toolLabelName = "Tool"
            if balanceCNNData:
                balancedDataset = self.createBalancedCNNDataset(fold)
                cnnTrainIndexes = self.loadData(fold,"Train",balancedDataset)
                cnnValIndexes = self.loadData(fold, "Validation",balancedDataset)
                cnnTestIndexes = self.loadData(fold, "Test",balancedDataset)
            else:
                cnnTrainIndexes = self.loadData(fold, "Train", self.dataCSVFile)
                cnnValIndexes = self.loadData(fold, "Validation", self.dataCSVFile)
                cnnTestIndexes = self.loadData(fold, "Test", self.dataCSVFile)
            self.batch_size = 8
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

            cnnLabelValues = numpy.array(sorted(self.dataCSVFile[toolLabelName].unique()))
            numpy.savetxt(os.path.join(foldDir,"cnn_labels.txt"),cnnLabelValues,fmt='%s',delimiter=',')


            cnnModel = network.createCNNModel((224,224,3),num_classes=len(cnnLabelValues))
            cnnModel.compile(optimizer = self.cnn_optimizer, loss = self.loss_Function, metrics = self.metrics)

            earlyStoppingCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
            modelCheckPointCallback = ModelCheckpoint(os.path.join(foldDir,'resnet50.h5'), verbose=1,monitor='val_accuracy', mode='max', save_weights_only = True,save_best_only=True)
            #self.cnnClassWeights = self.getClassWeights(cnnTrainDataSet.targets)

            self.cnnClassWeights = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1}

            history = cnnModel.fit(x=cnnTrainDataSet,
                                   validation_data=cnnValDataSet,
                                   epochs=self.numEpochs,callbacks=[modelCheckPointCallback])
            cnnModel.load_weights(os.path.join(foldDir, 'resnet50.h5'))

            results = cnnModel.evaluate(x=cnnTestDataSet)
            predictions = cnnModel.predict(cnnTestDataSet)
            predictions = numpy.argmax(predictions, axis=-1)
            trueLabels = numpy.argmax(cnnTestDataSet.targets, axis=-1)
            self.confMat = sklearn.metrics.confusion_matrix(trueLabels, predictions)
            print(self.confMat)
            self.saveTrainingInfo(fold, foldDir, history.history, results,"CNN")
            self.saveTrainingPlot(foldDir, history.history, "loss", "CNN")
            for metric in self.metrics:
                self.saveTrainingPlot(foldDir,history.history,metric,"CNN")
            #self.dataCSVFile = pandas.read_csv("D:/Datasets/Central_Line_Std_kit/TBME_2021_louo.csv")            lstmTrainIndexes = self.loadData(fold,"Train",self.dataCSVFile)
            lstmValIndexes = self.loadData(fold,"Validation",self.dataCSVFile)
            lstmTestIndexes = self.loadData(fold, "Test",self.dataCSVFile)
            
            lstmTrainSequences = self.splitDatasetIntoSequences(fold,"Train",sequenceLength=self.sequenceLength,downsampling=self.downsampling)
            lstmValSequences = self.splitDatasetIntoSequences(fold,"Validation",sequenceLength=self.sequenceLength,downsampling=self.downsampling)
            lstmTestSequences = self.splitDatasetIntoSequences(fold,"Test",sequenceLength=self.sequenceLength,downsampling=self.downsampling)

            self.lstmLabelValues = numpy.array(sorted(self.dataCSVFile[taskLabelName].unique()))
            numpy.savetxt(os.path.join(foldDir, "lstm_labels.txt"), self.lstmLabelValues, fmt='%s', delimiter=',')
            
            if balanceLSTMData:
                lstmTrainSequences = self.getBalancedSequences(lstmTrainSequences)
                lstmValSequences = self.getBalancedSequences(lstmValSequences)
            self.batch_size = 8
            lstmTrainDataSet = LSTMSequence(self.dataCSVFile, lstmTrainIndexes, lstmTrainSequences, cnnModel, self.batch_size, taskLabelName,tempFileDir)
            print("Training images loaded")
 
            lstmValDataSet = LSTMSequence(self.dataCSVFile, lstmValIndexes, lstmValSequences, cnnModel, self.batch_size, taskLabelName,tempFileDir)
            print("Validation images loaded")
            lstmTestDataSet = LSTMSequence(self.dataCSVFile, lstmTestIndexes, lstmTestSequences, cnnModel, self.batch_size, taskLabelName,tempFileDir)
            print("Test images loaded")


            #earlyStoppingCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=0)
            modelCheckPointCallback = ModelCheckpoint(os.path.join(foldDir, 'parallel_LSTM.h5'), verbose=1,
                                                      monitor='val_accuracy', mode='max', save_weights_only=True,
                                                      save_best_only=True)
            #earlyStoppingCallback = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3)
            #self.lstmClassWeights = self.getClassWeights(lstmTrainDataSet.targets)

            self.lstmClassWeights = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1}

            lstmModel = network.createLSTMModel(self.sequenceLength, numClasses=len(self.lstmLabelValues))
            lstmModel.compile(optimizer=self.lstm_optimizer, loss=self.loss_Function, metrics=self.metrics)
            history = lstmModel.fit(x=lstmTrainDataSet,
                                   validation_data=lstmValDataSet,
                                   epochs=self.numLSTMEpochs,callbacks=[modelCheckPointCallback],class_weight=self.lstmClassWeights)
            lstmModel.load_weights(os.path.join(foldDir, 'parallel_LSTM.h5'))
            results = lstmModel.evaluate(x=lstmTestDataSet)
            predictions = lstmModel.predict(lstmTestDataSet)
            predictions = numpy.argmax(predictions, axis=-1)
            trueLabels = numpy.argmax(lstmTestDataSet.targets, axis=-1)
            self.confMat = sklearn.metrics.confusion_matrix(trueLabels, predictions)
            print(self.confMat)
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
      default=20,
      help='number of epochs used in training'
  )
  parser.add_argument(
      '--num_epochs_lstm',
      type=int,
      default=20,
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
      default=8,
      help='type of output your model generates'
  )
  parser.add_argument(
      '--cnn_learning_rate',
      type=float,
      default=0.000001,
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
