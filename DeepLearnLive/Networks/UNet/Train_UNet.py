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
import UNet
from unetSequence import unetSequence

from tensorflow.keras import backend as K

FLAGS = None

class Train_UNet:

    #Loads the data from the specified CSV file
    # fold: The fold number given in the CSV file (should be an int)
    # set: which set the images and labels make up (should be one of: "Train","Validation", or "Test")
    # Returns:
    #   images: The list of images loaded from the files or girderIDs
    #   imageLabels: a dictionary of the labels for each image, indexed by the label name
    def loadData(self, fold, set, dataset):
        entries = dataset.loc[(dataset["Fold"] == fold) & (dataset["Set"] == set)]
        return entries.index

    def process_ultrasound(self, image):
        resized = cv2.resize(image, (128, 128)).astype(numpy.float16)
        scaled = resized / resized.max()
        return scaled[...,numpy.newaxis]

    def process_seg(self, image):
        resized = cv2.resize(image, (128, 128))
        return resized[...,numpy.newaxis] / 255

    def convertTextToNumericLabels(self,textLabels,labelValues):
        numericLabels =[]
        for i in range(len(textLabels)):
            label = numpy.zeros(len(labelValues))
            labelIndex = numpy.where(labelValues == textLabels[i])
            label[labelIndex] = 1
            numericLabels.append(label)
        return numpy.array(numericLabels)

    def saveTrainingInfo(self,foldNum,saveLocation,trainingHistory,results):
        LinesToWrite = []
        folds = "Fold " + str(foldNum) +"/"+ str(self.numFolds)
        modelType = "\nNetwork type: " + str(self.networkType)
        LinesToWrite.append(modelType)
        datacsv = "\nData CSV: " + str(FLAGS.data_csv_file)
        LinesToWrite.append(datacsv)
        numEpochs = "\nNumber of Epochs: " + str(self.numEpochs)
        LinesToWrite.append(numEpochs)
        batch_size = "\nBatch size: " + str(self.batch_size)
        LinesToWrite.append(batch_size)
        LearningRate = "\nLearning rate: " + str(self.learning_rate)
        LinesToWrite.append(LearningRate)
        LossFunction = "\nLoss function: " + str(self.loss_Function)
        LinesToWrite.append(LossFunction)
        trainStatsHeader = "\n\nTraining Statistics: "
        LinesToWrite.append(trainStatsHeader)
        trainLoss = "\n\tFinal training loss: " + str(trainingHistory["loss"][-1])
        LinesToWrite.append(trainLoss)
        for i in range(len(self.metrics)):
            trainMetrics = "\n\tFinal training " + self.metrics[i] + ": " + str(trainingHistory[self.metrics[i]][-1])
            LinesToWrite.append(trainMetrics)
        trainLoss = "\n\tFinal validation loss: " + str(trainingHistory["val_loss"][-1])
        LinesToWrite.append(trainLoss)
        for i in range(len(self.metrics)):
            valMetrics = "\n\tFinal validation " + self.metrics[i] + ": " + str(trainingHistory["val_"+self.metrics[i]][-1])
            LinesToWrite.append(valMetrics)
        testStatsHeader = "\n\nTesting Statistics: "
        LinesToWrite.append(testStatsHeader)
        testLoss = "\n\tTest loss: " + str(results[0])
        LinesToWrite.append(testLoss)
        for i in range(len(self.metrics)):
            testMetrics = "\n\tTest " + self.metrics[i] + ": " + str(results[i+1])
            LinesToWrite.append(testMetrics)

        with open(os.path.join(saveLocation,"trainingInfo.txt"),'w') as f:
            f.writelines(LinesToWrite)

    def saveTrainingPlot(self,saveLocation,history,metric):
        fig = plt.figure()
        plt.plot([x for x in range(self.numEpochs)], history[metric], 'bo', label='Training '+metric)
        plt.plot([x for x in range(self.numEpochs)], history["val_" + metric], 'b', label='Validation '+metric)
        plt.title('Training and Validation ' + metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(saveLocation, metric + '.png'))

    def train(self):
        self.saveLocation = FLAGS.save_location
        self.networkType = os.path.basename(os.path.dirname(self.saveLocation))
        self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
        self.numEpochs = 20
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_Function = IoU_loss
        self.metrics = ['IoU','accuracy']
        self.numFolds = self.dataCSVFile["Fold"].max() + 1
        self.gClient = None
        network = UNet.UNet()
        for fold in range(0,1): #changed from range(0,self.numFolds)
            foldDir = self.saveLocation+"_Fold_"+str(fold)
            os.mkdir(foldDir)
            labelName = "Tissue" #This should be the label that will be used to train the network

            trainIndexes = self.loadData(fold, "Train", self.dataCSVFile)
            valIndexes = self.loadData(fold, "Validation", self.dataCSVFile)
            testIndexes = self.loadData(fold, "Test", self.dataCSVFile)

            trainDataSet = unetSequence(self.dataCSVFile, trainIndexes, self.batch_size, labelName,self.gClient, tempFileDir)
            valDataSet = unetSequence(self.dataCSVFile, valIndexes, self.batch_size, labelName, self.gClient,tempFileDir)
            testDataSet = unetSequence(self.dataCSVFile, testIndexes, self.batch_size, labelName, self.gClient,tempFileDir)

            model = network.createModel((512,512,3),num_classes=10)
            print(model.summary())
            model.compile(optimizer = self.optimizer, loss = self.loss_Function, metrics = [IoU, 'accuracy'])
            history = model.fit(x=trainDataSet,
                                validation_data=valDataSet,
                                batch_size = self.batch_size,
                                epochs = self.numEpochs)

            results = model.evaluate(x = testDataSet,
                                     batch_size = self.batch_size)
            network.saveModel(model,foldDir)
            self.saveTrainingInfo(fold,foldDir,history.history,results)
            self.saveTrainingPlot(foldDir,history.history,"loss")
            for metric in self.metrics:
                self.saveTrainingPlot(foldDir,history.history,metric)

def IoU_loss(y_true,y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true[:,:,:,1] * y_pred[:,:,:,1])        #Create intersection
    sum_ = K.sum(y_true[:,:,:,1] + y_pred[:,:,:,1])                #Create union
    jac = (intersection + smooth) / (sum_ - intersection + smooth) #Divide and smooth
    return K.mean(1-jac) #Return 1-IoU so it can be use as a measurement of loss

def IoU(y_true,y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred[:,:,:,1], 0, 1))             #Extract binary mask from probability map
    intersection = K.sum(y_true[:,:,:,1] * y_pred_pos)              #Create union
    sum_ = K.sum(y_true[:,:,:,1] + y_pred[:,:,:,1])                 #Create intersection
    jac = (intersection + smooth) / (sum_ - intersection + smooth)  #Divide and smooth
    return K.mean(jac) #Return the mean jaccard index as IoU

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='C:/repos/aigt/DeepLearnLive/Networks/Vessel_UNet/EMBC/FullDataset1',
      help='Name of the directory where the models and results will be saved'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='C:/repos/aigt/DeepLearnLive/Datasets/US_Vessel_Segmentations/FullDataset_UNET.csv',
      help='Path to the csv file containing locations for all data used in training'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=40,
      help='number of epochs used in training'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='type of output your model generates'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.02,
      help='Learning rate used in training'
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
tm = Train_UNet()
tm.train()
