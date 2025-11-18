import numpy as np
import matplotlib.pyplot as plt
from medmnist import BloodMNIST

#Load data
dataset = BloodMNIST(split="val", download=True, size=64)

