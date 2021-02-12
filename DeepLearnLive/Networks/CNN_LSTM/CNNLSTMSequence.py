import numpy
import math
import pandas
import os
import cv2
import tensorflow
import tensorflow.keras
from tensorflow.keras.utils import Sequence
import girder_client


class CNNSequence(Sequence):
    def __init__(self,datacsv,indexes,batchSize,labelName,gClient = None,tempFileDir = None):
        # author Rebecca Hisey
        if "GirderID" in datacsv.columns:
            self.gClient = gClient
            self.tempFileDir = tempFileDir
            self.inputs = numpy.array([self.downloadGirderData(x,datacsv) for x in indexes])
        else:
            self.inputs = numpy.array([os.path.join(datacsv["Folder"][x],datacsv["FileName"][x]) for x in indexes])
        self.targets = numpy.array([datacsv[labelName][x] for x in indexes])
        self.batchSize = batchSize
        self.labelName = labelName
        self.labels = datacsv[self.labelName].unique()

    def __len__(self):
        # author Rebecca Hisey
        length = len(self.inputs) / self.batchSize
        length = math.ceil(length)
        return length

    def convertTextToNumericLabels(self, textLabel):
        label = numpy.zeros(len(self.labels))
        labelIndex = numpy.where(self.labels == textLabel)
        label[labelIndex] = 1
        return label

    def readImage(self,file):
        image = cv2.imread(file)
        resized_image = cv2.resize(image, (224, 224))
        return resized_image

    def downloadGirderData(self,index,datacsv):
        # tempFileDir is a folder in which to temporarily store the files downloaded from Girder
        # by default the temporary folder is created in the current working directory, but this can
        # be modified as necessary
        if not os.path.isdir(self.tempFileDir):
            os.mkdir(self.tempFileDir)
        fileID = datacsv["GirderID"][index]
        fileName = datacsv["FileName"][index]
        if not os.path.isfile(os.path.join(self.tempFileDir, fileName)):
            self.gClient.downloadItem(fileID, self.tempFileDir)
            numFilesWritten += 1
            if numFilesWritten % 100 == 0:
                print(numFilesWritten)
        return(os.path.join(self.tempFileDir, fileName))

    def __getitem__(self,index):
        # author Rebecca Hisey
        startIndex = index*self.batchSize
        indexOfNextBatch = (index + 1)*self.batchSize
        inputBatch = numpy.array([self.readImage(x) for x in self.inputs[startIndex : indexOfNextBatch]])
        outputBatch = numpy.array([self.convertTextToNumericLabels(x) for x in self.targets[startIndex : indexOfNextBatch]])
        return (inputBatch,outputBatch)


class LSTMSequence(Sequence):
    def __init__(self, datacsv, indexes, sequences, model, batchSize, labelName,tempFileDir = None):
        # author Rebecca Hisey
        self.cnnModel = model
        if tempFileDir == None:
            self.inputs = self.readImages([os.path.join(datacsv["Folder"][x], datacsv["FileName"][x]) for x in indexes])
        else:
            self.inputs = self.readImages([os.path.join(tempFileDir, datacsv["FileName"][x]) for x in indexes])
        self.targets = numpy.array([datacsv[labelName][x] for x in indexes])
        self.sequences = sequences
        self.batchSize = batchSize
        self.labelName = labelName
        self.labels = datacsv[self.labelName].unique()
        inputSequences, targetSequences = self.readImageSequences()
        self.inputs = inputSequences
        self.targets = targetSequences

    def __len__(self):
        # author Rebecca Hisey
        length = len(self.inputs) / self.batchSize
        length = math.ceil(length)
        return length

    def readImages(self,files):
        images = []
        numLoaded = 0
        for file in files:
            image = cv2.imread(file)
            resized_image = cv2.resize(image, (224, 224))
            images.append(resized_image)
            numLoaded +=1
            if numLoaded % 1000 == 0:
                print("loaded " +str(numLoaded)+' / '+str(len(files))+ ' images')
        cnnOutput = self.cnnModel.predict(numpy.array(images))
        return cnnOutput

    def getSequenceLabels(self, sequence):
        textLabel = self.targets[sequence[len(sequence)-1]]
        label = self.convertTextToNumericLabels(textLabel)
        return numpy.array(label)

    def convertTextToNumericLabels(self, textLabel):
        label = numpy.zeros(len(self.labels))
        labelIndex = numpy.where(self.labels == textLabel)
        label[labelIndex] = 1
        return label

    def readImageSequences(self):
        allSequences = []
        allLabels = []
        for sequence in self.sequences:
            predictedSequence = []
            label = self.getSequenceLabels(sequence)
            for i in range(len(sequence)):
                image = self.inputs[sequence[i]]
                predictedSequence.append(image)
            if predictedSequence != []:
                allSequences.append(predictedSequence)
                allLabels.append(label)
        return(numpy.array(allSequences),numpy.array(allLabels))

    def __getitem__(self, index):
        # author Rebecca Hisey
        startIndex = index * self.batchSize
        indexOfNextBatch = (index + 1) * self.batchSize
        inputBatch = numpy.array([x for x in self.inputs[startIndex: indexOfNextBatch]])
        outputBatch = numpy.array([x for x in self.targets[startIndex: indexOfNextBatch]])
        if inputBatch.shape == (0,) or outputBatch.shape == (0,):
            print(inputBatch.shape)
            print(self.sequences[startIndex: indexOfNextBatch])
            print(outputBatch.shape)
            print(inputBatch)
            print(outputBatch)
        return (inputBatch, outputBatch)
