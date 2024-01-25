import akshare as ak
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from torcheval.metrics import BinaryAccuracy

metric = BinaryAccuracy()
input = torch.tensor([0.0, 0.0, 0.1, 0.1])
target = torch.tensor([1.0, 0.0, 1.0, 1.0])
metric.update(input, target)
a = metric.compute()

print(a)
metric.reset()
metric.update(torch.tensor([1.0]), torch.tensor([1.0]))

print(metric.compute())
