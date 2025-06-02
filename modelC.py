import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

def build_densenet(num_classes):
    weights = DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model
