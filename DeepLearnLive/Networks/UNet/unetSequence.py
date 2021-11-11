import numpy
import math
import os
import gc
import cv2
import pandas
from tensorflow.keras.utils import Sequence,to_categorical
import girder_client
from sklearn.utils import shuffle
import random


class unetSequence(Sequence):
    def __init__(self,datacsv,indexes,batchSize,labelName,gClient = None,tempFileDir = None,shuffle=True,augmentations = False,testSet=False):
        # author Rebecca Hisey
        if "GirderID" in datacsv.columns:
            self.gClient = gClient
            self.tempFileDir = tempFileDir
            self.inputs = numpy.array([self.downloadGirderData(x,datacsv) for x in indexes])
        else:
            self.inputs = numpy.array([os.path.join(datacsv["Folder"][x],datacsv["FileName"][x]) for x in indexes])
            self.targets = numpy.array([os.path.join(datacsv["Folder"][x],datacsv[labelName][x]) for x in indexes])
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.testSet=testSet
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            shuffledInputs,shuffledTargets = shuffle(self.inputs,self.targets)
            self.inputs = shuffledInputs
            self.targets = shuffledTargets
        gc.collect()


    def __len__(self):
        # author Rebecca Hisey
        length = len(self.inputs) / self.batchSize
        length = math.ceil(length)
        return length

    def rotateImage(self,image,angle = -1):
        if angle < 0:
            angle = random.randint(1, 359)
        center = tuple(numpy.array(image.shape[1::-1])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        rotImage = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_LINEAR)
        return rotImage

    def flipImage(self,image,axis):
        return cv2.flip(image, axis)

    def readImage(self,RGBfile):
        image = cv2.imread(RGBfile)

        resized = cv2.resize(image, (512,512)).astype(numpy.float16)
        scaled = resized / resized.max()
        return scaled

    def readSegmentationImage(self,segFile):
        image = cv2.imread(segFile)
        resized = cv2.resize(image, (512,512))
        resized = resized[...,numpy.newaxis] / 255
        #seg_train_onehot = to_categorical(resized, 10)
        return seg_train_onehot


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

        inputBatch = [self.readImage(x) for x in self.inputs[startIndex:indexOfNextBatch]]

        outputBatch = [self.readSegmentationImage(x) for x in self.targets[startIndex : indexOfNextBatch]]

        inputBatch = numpy.array(inputBatch)
        outputBatch = numpy.array(outputBatch)
        return (inputBatch,outputBatch)