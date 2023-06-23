import argparse
import os
import platform
import sys
from pathlib import Path

import pandas
import torch
import numpy as np
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import pandas

class Rebecca_YOLOv5:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_det = 1000
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.01
        self.agnostic_nms = False
        self.model = None

    def loadModel(self, modelFolder, modelName):
        weights = os.path.join(modelFolder,"weights","best.pt")
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=None, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def predict(self,image):
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,0)
        im = image.copy()
        im = np.transpose(im, [2, 0, 1])
        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        print(im.shape)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim


        print(im.shape)
        # Inference
        pred = self.model(im, augment=False)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, self.agnostic_nms, max_det=self.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            bboxes = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()

                # Write results

                for *xyxy, conf, cls in reversed(det):
                    bbox = {}
                    bbox["class"] = self.names[int(cls.item())]
                    bbox["xmin"] = int(xyxy[0].item())
                    bbox["xmax"] = int(xyxy[2].item())
                    bbox["ymin"] = int(xyxy[1].item())
                    bbox["ymax"] = int(xyxy[3].item())
                    bbox["conf"] = float(conf.item())
                    bboxes.append(bbox)
                    image = cv2.rectangle(image,(bbox["xmin"],bbox["ymin"]),(bbox["xmax"],bbox["ymax"]),(255,0,0),2)
        print(image.shape)
        #image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image,axis=0)
        return image#str(bboxes)

