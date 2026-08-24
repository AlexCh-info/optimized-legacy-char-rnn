from model import *
import torch

decoder = decoder = torch.load("choto.pt", weights_only=False)
print("Model loaded")