import os
import cv2
import sys
import numpy
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

class CNN_LSTM():
    def __init__(self):
        self.cnnModel = None
        self.lstmModel = None
        self.imageSequence = numpy.zeros((50,8))

    def loadModel(self,modelFolder,modelName):
        self.cnnModel = self.loadCNNModel(modelFolder,'cnn_'+modelName)
        self.lstmModel = self.loadLSTMModel(modelFolder,'lstm_' + modelName)

    def loadCNNModel(self,modelFolder, modelName):
        structureFileName = modelName + '.json'
        weightsFileName = modelName + '.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        model = model_from_json(JSONModel)
        model.load_weights(os.path.join(modelFolder, weightsFileName))
        #model.compile()
        return model

    def loadLSTMModel(self,modelFolder, modelName):
        structureFileName = modelName + '.json'
        weightsFileName = modelName + '.h5'
        modelFolder = modelFolder.replace("'", "")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        model = model_from_json(JSONModel)
        model.load_weights(os.path.join(modelFolder, weightsFileName))
        adam = tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = ['anesthetic', 'dilator', 'insert_catheter', 'insert_guidewire', 'insert_needle', 'nothing','remove_guidewire', 'scalpel']
        resized = cv2.resize(image, (224, 224))
        resized = numpy.expand_dims(resized, axis=0)
        toolClassification = self.cnnModel.predict(numpy.array(resized))
        self.imageSequence = numpy.append(self.imageSequence[1:], toolClassification, axis=0)
        taskClassification = self.lstmModel.predict(numpy.array([self.imageSequence]))
        labelIndex = numpy.argmax(taskClassification)
        label = labels[labelIndex]
        networkOutput = str(label) + str(taskClassification)
        return networkOutput