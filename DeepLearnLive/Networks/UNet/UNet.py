import os
import cv2
import sys
import numpy
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

sys.path.append('C:\\repos\\aigt')

import segmentation_unet as unet

''' 
Sample Model file for defining a neural network for use within the DeepLearnLive extension
Originally developed by Rebecca Hisey for the Laboratory of Percutaneous Surgery, Queens University, Kingston, ON

Model description: 
    Include a description of the intended use of your model here.
    This example shows a simple CNN that predicts tools from RGB images and returns a string
'''

class UNet():
    def __init__(self):
        self.unetModel = None
        self.labels = None

        #Unet definition parameters
        self.filter_multiplier = 8
        self.reg_rate = 0.0001

    def loadModel(self,modelFolder,modelName):
        #Replace the following lines with whatever needs to be done to load the model or models
        structureFileName = 'unet.json'
        weightsFileName = 'unet.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        self.cnnModel = model_from_json(JSONModel)
        self.cnnModel.load_weights(os.path.join(modelFolder, weightsFileName))
        with open(os.path.join(modelFolder,"labels.txt"),'r') as f:
            self.labels = f.read()
        self.labels = self.labels.split(sep="\n")

    def predict(self,image):
        #Replace the following lines with whatever needs to be done to use the model to predict on new data
        # in this case the image needed to be recoloured and resized and our prediction returns the tool name and the
        # softmax output
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224))
        resized = numpy.expand_dims(resized, axis=0)
        toolClassification = self.cnnModel.predict(numpy.array(resized))
        labelIndex = numpy.argmax(toolClassification)
        label = self.labels[labelIndex]
        networkOutput = str(label) + str(toolClassification)
        return networkOutput

    def createModel(self,imageSize,num_classes):
        model = unet.segmentation_unet(imageSize[0], num_classes, self.filter_multiplier, self.reg_rate)
        return model

    def saveModel(self,trainedModel,saveLocation):
        JSONmodel = trainedModel.to_json()
        structureFileName = 'unet.json'
        weightsFileName = 'unet.h5'
        with open(os.path.join(saveLocation,structureFileName),"w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)
        trainedModel.save_weights(os.path.join(saveLocation,weightsFileName))