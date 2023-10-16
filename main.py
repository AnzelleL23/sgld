import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import PIL

import sys
import os

import sgld

train_loader, test_loader = sgld.make_datasets()
sgld.visualise_samples(train_loader, 5)

torch.zeros(100)