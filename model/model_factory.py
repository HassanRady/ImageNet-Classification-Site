import torch
from torch import cuda
from torchvision import transforms, models
from PIL import Image
import json

from model.model import Model
import  model.config as config

class ModelFactory(object):
    def __init__(self):
       pass

    def get_resnet34(self, save=False):
        model = models.resnet34(pretrained=True)
        model_name = model._get_name()

        model.eval()

        if save:
            model_path = config.MODEL_TRAINED_DIR / f"{model_name}.pth"
            torch.save(model, model_path)

        model = Model(model)
        return model

