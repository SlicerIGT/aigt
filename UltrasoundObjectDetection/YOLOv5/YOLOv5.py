import sys
import torch
import sys
import numpy as np

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

sys.path.append('C:\\repos\\aigt')

''' 
Sample Model file for defining a neural network for use within the DeepLearnLive extension
Originally developed by Rebecca Hisey for the Laboratory of Percutaneous Surgery, Queens University, Kingston, ON

Model description: 
    Include a description of the intended use of your model here.
    This example shows a simple CNN that predicts tools from RGB images and returns a string
'''

class YOLOv5():
    def __init__(self):
        self.model = None
        self.class_names = None
        self.line_thickness = 1

    def loadModel(self,modelFolder,modelName):
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weights = f'{modelFolder}/weights/best.pt'

        self.model = DetectMultiBackend(weights=weights, device=device)
        self.class_names = self.model.names

    def predict(self,image):
        #Replace the following lines with whatever needs to be done to use the model to predict on new data
        # in this case the image needed to be recoloured and resized and our prediction returns the tool name and the
        # softmax output
        im = image.copy()
        im = np.stack([im, im, im], axis=-1)
        
        im = torch.from_numpy(im).permute(2,0,1).to(self.model.device)

        
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, visualize=False)

        pred = non_max_suppression(pred)

        for det in pred:  # per image
            im0 = np.ascontiguousarray(image.copy())

            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.class_names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = (f'{self.class_names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        return im0


    def createModel(self,imageSize,num_classes):
        pass

    def saveModel(self,trainedModel,saveLocation):
        pass