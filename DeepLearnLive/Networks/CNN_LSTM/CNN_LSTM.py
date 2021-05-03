import os
import cv2
import sys
import numpy
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import InceptionV3,ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

class CNN_LSTM():
    def __init__(self):
        self.cnnModel = None
        self.lstmModel = None
        self.imageSequence = numpy.zeros((50,8))

    def loadModel(self,modelFolder,modelName):
        self.cnnModel = self.loadCNNModel(modelFolder)
        self.lstmModel = self.loadLSTMModel(modelFolder)

    def loadCNNModel(self,modelFolder):
        structureFileName = 'resnet50.json'
        weightsFileName = 'resnet50.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        model = model_from_json(JSONModel)
        model.load_weights(os.path.join(modelFolder, weightsFileName))
        #model.compile()
        return model

    def loadLSTMModel(self,modelFolder):
        structureFileName = 'parallel_LSTM.json'
        weightsFileName = 'parallel_LSTM.h5'
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

        #flipped = cv2.flip(image, 0)
        labels = ['anesthetic', 'dilator', 'insert_catheter', 'insert_guidewire', 'insert_needle', 'nothing','remove_guidewire', 'scalpel']
        toolLabels = ['anesthetic', 'catheter', 'dilator', 'guidewire','guidewire_casing', 'nothing', 'scalpel', 'syringe']
        resized = cv2.resize(image, (224, 224)) #MobileNet
        #resized = cv2.resize(image, (299, 299))  #InceptionV3
        normImage = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #cv2.imshow("norm", normImage)
        #cv2.waitKey(0)
        normImage = numpy.expand_dims(normImage, axis=0)

        toolClassification = self.cnnModel.predict(numpy.array(normImage))
        print(toolLabels[numpy.argmax(toolClassification)])
        self.imageSequence = numpy.append(self.imageSequence[1:], toolClassification, axis=0)
        taskClassification = self.lstmModel.predict(numpy.array([self.imageSequence]))
        labelIndex = numpy.argmax(taskClassification)
        label = labels[labelIndex]
        networkOutput = str(label) + str(taskClassification)
        return networkOutput

    def createCNNModel(self,imageSize,num_classes):
        #Replace the following lines with your model definition
        # in this example we create a MobileNetV2 model and initialize the model with weights from training on ImageNet
        model = tensorflow.keras.models.Sequential()
        model.add(ResNet50(weights='imagenet',include_top=False,input_shape=imageSize,pooling='avg'))
        #model.add(InceptionV3(weights='imagenet', include_top=False, input_shape=imageSize))
        model.add(layers.Dense(512,activation='relu'))
        model.add(layers.Dense(num_classes,activation='softmax'))
        return model

    def createLSTMModel(self,sequenceLength, numClasses):
        input = layers.Input(shape=(sequenceLength, numClasses))
        # model = tensorflow.keras.models.Sequential()

        bLSTM0 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        bLSTM1 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        bLSTM2 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        bLSTM3 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        bLSTM4 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        bLSTM5 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        bLSTM6 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        bLSTM7 = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        d1 = layers.Concatenate(axis=1)([bLSTM0, bLSTM1, bLSTM2, bLSTM3, bLSTM4, bLSTM5, bLSTM6, bLSTM7])
        r1 = layers.Dense(numClasses, activation='relu')(d1)
        r2 = layers.Dense(numClasses, activation='relu')(r1)
        out = layers.Dense(numClasses, activation='softmax')(r2)
        model = tensorflow.keras.models.Model(inputs=input, outputs=out)
        adam = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def createSingleLSTMModel(self,sequenceLength, numClasses):
        input = layers.Input(shape=(sequenceLength, numClasses))
        # model = tensorflow.keras.models.Sequential()
        bLSTM = layers.Bidirectional(layers.LSTM(numClasses, return_sequences=False))(input)
        r1 = layers.Dense(numClasses, activation='relu')(bLSTM)
        r2 = layers.Dense(numClasses, activation='relu')(r1)
        out = layers.Dense(numClasses, activation='softmax')(r2)
        model = tensorflow.keras.models.Model(inputs=input, outputs=out)
        adam = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def saveModel(self,trainedCNNModel,trainedLSTMModel,saveLocation):
        JSONmodel = trainedCNNModel.to_json()
        structureFileName = 'resnet50.json'
        weightsFileName = 'resnet50.h5'
        #structureFileName = 'inceptionv3.json'
        #weightsFileName = 'inceptionv3.h5'
        with open(os.path.join(saveLocation,structureFileName),"w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)
        #trainedCNNModel.save_weights(os.path.join(saveLocation,weightsFileName))

        JSONmodel = trainedLSTMModel.to_json()
        structureFileName = 'parallel_LSTM.json'
        weightsFileName = 'parallel_LSTM.h5'
        with open(os.path.join(saveLocation, structureFileName), "w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)
        #trainedLSTMModel.save_weights(os.path.join(saveLocation, weightsFileName))