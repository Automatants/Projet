import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
from torchvision.transforms import Resize, functional

import cv2 as cv

from networks.pix2pix import Pix2Pix

PATH_TO_DATA = "./dataset/A"
PATH_TO_RESULT = "./dataset/B"
PATH_TO_MODEL = "./checkpoints/generator_gan128_weights"

pix2pix = Pix2Pix(0.0002, 0.5, 0.5, test_batch=None)
pix2pix.generator = torch.load(PATH_TO_MODEL)
pix2pix.generator.eval()

SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
 
size_list = []
name_list = []
dataset = []

for file in os.listdir(PATH_TO_DATA):
    image = cv.imread(os.path.join(PATH_TO_DATA, file))
    background = cv.morphologyEx(image, cv.MORPH_DILATE, SE)
    image = cv.divide(image, background, scale=255)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(os.path.join(PATH_TO_DATA,"n" + file), image)
    tensor = torchvision.io.read_image(os.path.join(PATH_TO_DATA,"n"+file))
    tensor = tensor/127.5 - 1
    input_shape = tensor.shape
    size_list.append(input_shape)
    name_list.append(file)
    tensor_shape = [(input_shape[i]//32)*32 for i in range(len(input_shape))]
    tensor = Resize(tensor_shape[1:])(tensor)
    #tensor = Resize([256, 256])(tensor)
    
    dataset.append(tensor)
    
#dataset = torch.stack(dataset)
#dataset = pix2pix.generator(dataset)

for i, tensor in enumerate(dataset):
    tensor = pix2pix.generator(tensor)
    print(i, tensor.shape)
    input_shape = size_list[i]
    tensor = Resize(input_shape[1:])(tensor)
    tensor = (tensor+1)/2
    tensor = functional.convert_image_dtype(tensor, dtype=torch.uint8)
    print(tensor.shape)
    torchvision.io.write_png(tensor, os.path.join(PATH_TO_RESULT, name_list[i]))

for file in name_list:
    os.remove(os.path.join(PATH_TO_DATA, "n" + file))
    