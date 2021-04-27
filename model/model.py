import torch
from torch import cuda
from torchvision import transforms, models
from PIL import Image
import json


class Model(object):
    def __init__(self, model):
        self._model = model
        self._device = torch.device("cuda" if cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self._imgnet_json = json.load(open('imagenet_class_index.json'))

    @property
    def device(self):
        return self._device

    @property
    def transform(self):
        return self._transform

    @property
    def imgnet_json(self):
        return self._imgnet_json

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model

    def __str__(self):
        return f"{self._model._get_name()} Model"

    def predict(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        else:
            img = Image.fromarray(img)

        img_processd = self.transform(img)
        img_processd = img_processd.unsqueeze(dim=0)
        img_processd = img_processd.to(self.device)

        output = self._model(img_processd)
        pred = torch.argmax(output).item()
        pred_class = self.imgnet_json[str(pred)][1]

        return pred_class
