import torch
import PIL
# Import resources
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import time
import json
import copy

import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import PIL

from PIL import Image
from collections import OrderedDict


import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

import os
import json

import training


def test():
    image_path = 'D:\\snake_data\\validate\\class-78\\65.jpg'
    img = Image.open(image_path)
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # Process a PIL image for use in a PyTorch model
        # tensor.numpy().transpose(1, 2, 0)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        image = preprocess(image)
        return image
    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax
    with Image.open('D:\\snake_data\\validate\\class-78\\65.jpg') as image:
        plt.imshow(image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = train.model
        
    def predict2(image_path, model, topk=5):
##        ''' Predict the class (or classes) of an image using a trained deep learning model.
##        '''
##        
        # Implement the code to predict the class from an image file
        img = Image.open(image_path)
        img = process_image(img)
        
        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)
        
        
        img = torch.from_numpy(img)
        
        train.model.eval()
        inputs = Variable(img).to(device)
        logits = model.forward(inputs)
        
        ps = F.softmax(logits,dim=1)
        topk = ps.cpu().topk(topk)
        
        return (e.data.numpy().squeeze().tolist() for e in topk)

    img_path = 'D:\snake_data\\validate\\class-78\\65.jpg'
    probs, classes = predict2(img_path, model.to(device))
    print(probs)
    print(classes)
    snake_names = [snake_class[class_names[e]] for e in classes]
    print(snake_names)

    def view_classify(img_path, prob, classes, mapping):
##        ''' Function for viewing an image and it's predicted classes.
##        '''
        image = Image.open(img_path)

        fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
        snake_name = mapping[img_path.split('/')[-2]]
        ax1.set_title(snake_name)
        ax1.imshow(image)
        ax1.axis('off')
        
        y_pos = np.arange(len(prob))
        ax2.barh(y_pos, prob, align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(snake_names)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_title('Class Probability')

    view_classify(img_path, probs, classes, snake_class)

if __name__ == '__main__':

    training.train()
    test()
