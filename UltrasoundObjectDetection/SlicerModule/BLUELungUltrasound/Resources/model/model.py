import torch
import numpy as np
import sys
from pathlib import Path
from torchvision.transforms import Resize, Compose, CenterCrop

try:
    from monai.visualize.class_activation_maps import GradCAMpp
except ImportError:
    slicer.util.pip_install('monai')
    from monai.visualize.class_activation_maps import GradCAMpp

RESIZED_SIZE = 256
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


CLASS_NAMES = {
    0: 'barcode',
    1: 'lung point',
    2: 'lung pulse',
    3: 'seashore'
}


class EnsembleClassifier(torch.nn.Module):
    def __init__(self, models, target_layers = None):
        super().__init__()
        self.models = models
        
        if target_layers:
            self.gradcams = [GradCAMpp(model, target_layer) for model, target_layer in zip(models, target_layers)]
        
        self.transforms = Compose([
            Resize(size=(RESIZED_SIZE, RESIZED_SIZE), antialias=True),
            CenterCrop(size=(RESIZED_SIZE, RESIZED_SIZE))
        ])
    
    def forward(self, x):
        return [model(x) for model in self.models]

    def get_gradcams(self, x):
        if self.gradcams:
            return [1-gradcam(x) for gradcam in self.gradcams]        

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            individual_preds = self.forward(self.transforms(x))

        probabilities = np.stack([torch.softmax(pred, dim=1).cpu().numpy() for pred in individual_preds], axis=0)
        ensemble_model_pred = probabilities.mean(axis=0).argmax(axis=1)
        return CLASS_NAMES[ensemble_model_pred.item()]