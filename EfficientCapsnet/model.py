from layers import DigitCap
from layers import FeatureMap
from layers import PrimaryCap
from losses import MarginLoss
from param import CapsNetParam

from torch.optim import Adam
import torchmetrics
import os
import torch
from torch import nn
from typing import List
from typing import Union

def make_model(
    param: CapsNetParam,
    optimizer = None,
    loss=nn.MarginRankingLoss(),
    metrics:torchmetrics = ["accuracy"]
):
    input_images =nn.Input(
        shape=[param.input_height, param.input_width, param.input_channel],
        name="input_images")
    feature_maps = FeatureMap(param, name="feature_maps")(input_images)
    primary_caps = PrimaryCap(param, name="primary_caps")(feature_maps)
    digit_caps = DigitCap(param, name="digit_caps")(primary_caps)
    digit_probs = nn.Lambda(lambda x: torch.norm(x, axis=-1),
                                         name="digit_probs")(digit_caps)

    model = nn.Model(inputs=input_images,
                           outputs=digit_probs,
                           name="Efficient-CapsNet")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

