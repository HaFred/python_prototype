import torch
import math
from collections import defaultdict
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data.dataloader as dataloader
import time

from neural_network import NeuralNetwork
from my_img2num import MyImg2Num

def main ():
  model_my = MyImg2Num()
  model_my.train()
  print('done')

if __name__ == "__main__":
  main()