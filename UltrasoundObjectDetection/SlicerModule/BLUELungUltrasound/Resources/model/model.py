import torch
import numpy as np
import sys
import yaml
from pathlib import Path
from torchvision.transforms import Resize, ToTensor, Compose, Pad

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

class ObjectDetectionModel():
    def __init__(self,
                 input_size,
                 model='weights/lung_us_pretrained.torchscript',
                 data_yaml='weights/lung_us.yml',
                 device=torch.device('cpu'),
                 line_thickness=2,
                 target_size=512):
        
        self.model = torch.jit.load(model if Path(model).is_absolute() else f'{str(ROOT)}/{model}', map_location=device)
        self.class_names = self._get_class_names_from_yaml(data_yaml if Path(data_yaml).is_absolute() else f'{str(ROOT)}/{data_yaml}')
        self.line_thickness = line_thickness
        self.device = device
        self.input_transform = Compose([
            ToTensor(),
            Pad(padding=self._get_padding_values(input_size=input_size)),
            Resize(size=target_size)
        ])

    def predict(self, image, confidence_threshold=0.5):
        #self.model.to(self.device)
        im = image.copy()
        im = self.input_transform(im).to(self.device).type(torch.float)
        if len(im.shape) == 3:
            im = torch.unsqueeze(im, dim=0)

        pred = self.model(im)
        pred = non_max_suppression(pred, conf_thres=confidence_threshold)

        im0 = np.ascontiguousarray(image.copy())
        for det in pred:  # per image

            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.class_names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = (f'{self.class_names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
        im0 = np.expand_dims(im0, axis=0)
        return im0
    
    def _get_padding_values(self, input_size):
        height, width = input_size
        difference = width - height
        left, top, right, bottom = 0, 0, 0, 0
        pad_value = difference // 2
        if difference > 0:
            top = pad_value
            bottom = pad_value if difference % 2 == 0 else pad_value + 1
        if difference < 0:
            left = pad_value
            right = pad_value if difference % 2 == 0 else pad_value + 1
        return left, top, right, bottom
    
    def _get_class_names_from_yaml(self, yaml_path):
        with open(yaml_path, 'r') as stream:
            parsed_yaml = yaml.safe_load(stream)
            class_names = parsed_yaml['names']
            return class_names