import numpy
import math
import os
import gc
import cv2
import pandas
from tensorflow.keras.utils import Sequence
import girder_client
from sklearn.utils import shuffle
import random


class CNNSequence(Sequence):
    def __init__(self,datacsv,indexes,batchSize,labelName,gClient = None,tempFileDir = None,shuffle=True,augmentations = True):
        # author Rebecca Hisey
        if "GirderID" in datacsv.columns:
            self.gClient = gClient
            self.tempFileDir = tempFileDir
            self.inputs = numpy.array([self.downloadGirderData(x,datacsv) for x in indexes])
        else:
            self.inputs = numpy.array([os.path.join(datacsv["Folder"][x],datacsv["FileName"][x]) for x in indexes])
        self.batchSize = batchSize
        self.labelName = labelName
        self.labels = numpy.array(sorted(datacsv[self.labelName].unique()))
        self.targets = numpy.array([self.convertTextToNumericLabels(datacsv[labelName][x]) for x in indexes])
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            shuffledInputs,shuffledTargets = shuffle(self.inputs,self.targets)
            self.inputs = shuffledInputs
            self.targets = shuffledTargets


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

    def rotateImage(self,image,angle):
        center = tuple(numpy.array(image.shape[1::-1])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        rotImage = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_LINEAR)
        return rotImage

    def readImage(self,file):
        image = cv2.imread(file)
        try:
            resized_image = cv2.resize(image, (224, 224))
            normImg = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            preprocessingMethod = random.randint(0, 3)
            if self.augmentations and preprocessingMethod == 0:
                # flip along y axis
                return cv2.flip(normImg, 1)
            if self.augmentations and preprocessingMethod == 1:
                # flip along x axis
                return cv2.flip(normImg, 0)
            elif self.augmentations and preprocessingMethod == 2:
                # rotate
                angle = random.randint(1, 359)
                rotImage = self.rotateImage(normImg, angle)
                return rotImage
            else:
                return normImg
        except:
            print(file)
        #resized_image = cv2.resize(image, (299, 299))


    def downloadGirderData(self,index,datacsv):
        # tempFileDir is a folder in which to temporarily store the files downloaded from Girder
        # by default the temporary folder is created in the current working directory, but this can
        # be modified as necessary
        if not os.path.isdir(self.tempFileDir):
            os.mkdir(self.tempFileDir)
        fileID = datacsv["GirderID"][index]
        fileName = datacsv["FileName"][index]
        numFilesWritten = 0
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
        outputBatch = numpy.array([x for x in self.targets[startIndex : indexOfNextBatch]])
        return (inputBatch,outputBatch)


class LSTMSequence(Sequence):
    def __init__(self, datacsv, indexes, sequences, model, batchSize, labelName,tempFileDir = None,shuffle=True):
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
        self.labels = numpy.array(sorted(datacsv[self.labelName].unique()))
        inputSequences, targetSequences = self.readImageSequences(indexes)
        self.inputs = inputSequences
        self.targets = targetSequences
        print('Class counts:' + str(numpy.sum(self.targets,axis=0)))
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            shuffledInputs,shuffledTargets = shuffle(self.inputs,self.targets)
            self.inputs = shuffledInputs
            self.targets = shuffledTargets

    def __len__(self):
        # author Rebecca Hisey
        length = len(self.inputs) / self.batchSize
        length = math.ceil(length)
        return length

    def readImages(self, files):
        images = []
        numLoaded = 0
        allOutputs = numpy.array([])
        for i in range(len(files)):
            image = cv2.imread(files[i])
            resized_image = cv2.resize(image, (224, 224))
            normImg = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(normImg)
            numLoaded += 1
            if numLoaded % 500 == 0 or i == (len(files) - 1):
                print("loaded " + str(numLoaded) + ' / ' + str(len(files)) + ' images')
                if allOutputs.size == 0:
                    allOutputs = self.cnnModel.predict(numpy.array(images))
                else:
                    cnnOutput = self.cnnModel.predict(numpy.array(images))
                    allOutputs = numpy.append(allOutputs, cnnOutput, axis=0)
                del images
                gc.collect()
                images = []
            del image
            del resized_image
            del normImg
        return allOutputs

    def getSequenceLabels(self, sequence,smallestIndex):
        '''if sequence[len(sequence)-1] >= len(self.targets):
            textLabel = self.targets[-1]
        else:'''
        try:
            textLabel = self.targets[sequence[-1]-smallestIndex]
            label = self.convertTextToNumericLabels(textLabel)
            return numpy.array(label)
        except IndexError:
            print(sequence)

    def convertTextToNumericLabels(self, textLabel):
        label = numpy.zeros(len(self.labels))
        labelIndex = numpy.where(self.labels == textLabel)
        label[labelIndex] = 1
        return label

    def readImageSequences(self,indexes):
        allSequences = []
        allLabels = []
        smallestIndex = indexes[0]
        for sequence in self.sequences:
            predictedSequence = []
            label = self.getSequenceLabels(sequence,smallestIndex)
            for i in range(len(sequence)):
                if sequence[i] == -1:
                    image = numpy.zeros(self.inputs[0].shape)
                    labelIndex = numpy.where(self.labels == "nothing")
                    image[labelIndex] = 1.0
                else:
                    image = self.inputs[sequence[i]-smallestIndex]
                predictedSequence.append(image)
            if predictedSequence != []:
                allSequences.append(predictedSequence)
                allLabels.append(label)
        return (numpy.array(allSequences), numpy.array(allLabels))

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
