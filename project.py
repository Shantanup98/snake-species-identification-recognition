from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2

import time
import json
import copy

import matplotlib.pyplot as plt
import seaborn as sns
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

import csv

import itertools

root = Tk()  # interface



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = 'D:\data'
train_dir = data_dir + '\train'
valid_dir = data_dir + '\validate'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
}

# Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validate']}

# Using the image datasets and the trainforms, define the dataloaders
batch_size = 10
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
          for x in ['train', 'validate']}

class_names = image_datasets['train'].classes

##print(class_names)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}
class_names = image_datasets['train'].classes

##print(dataset_sizes)
#print(device)

# Label mapping
with open('D:\\snake\data.json', 'r') as f:
    snake_class = json.load(f)





# Loading the checkpoint
ckpt = torch.load('D:\\snake\\checkpoint_ic_d161.pth', map_location=device)
ckpt.keys()


# Load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']


model, class_to_idx = load_checkpoint('D:\\snake\\checkpoint_ic_d161.pth')


idx_to_class = { v : k for k,v in class_to_idx.items()}


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.class_to_idx = image_datasets['train'].class_to_idx


def predict2(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    
    
    img = torch.from_numpy(img)
    
    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

def view_classify(img_path, prob, classes, top_name, snake_names):
    ''' Function for viewing an image and it's predicted classes.
    '''
    image = Image.open(img_path)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    names = snake_names
##    snake_name = mapping[img_path.split('\\')[-2]]
    snake_name = top_name
    print(snake_name)
    ax1.set_title(snake_name)
    ax1.imshow(image)
    ax1.axis('off')
    
    y_pos = np.arange(len(prob))
    ax2.barh(y_pos, prob, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_title('Class Probability')

    plt.show()



def open_img():
    # Select the Imagename from a folder
    global x
    x = openfilename()

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((300, 250))  # Image.ANTIALIAS)

    # PhotoImage class is used to display image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    panel.place(x=70, y=225)


def openfilename():
    global img
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    img = cv2.imread(filename, 0)
    return filename


##def getpath():
##    a=x
##    s1.set(a)
##    #return a

def predict():
    global img_path, probs, classes, top_name, top_classes, classid, sname, cname, ven
    img_path = x
    print(img_path)
    probs, classes = predict2(img_path, model.to(device))
    print(probs)
    print(classes)
    top_classes = classes[0]
    print(top_classes)
    snake_names = [snake_class[class_names[e]] for e in classes]
    top_name = snake_names[0]


##    with open('G:\\BE proj\\class_info.csv', 'r') as f:
##        row = next(itertools.islice(csv.reader(f), top_classes+1, None))
##        classid = row[0]
##        sname = row[1]
##        ven = row[2]
##        cname = row[3]


    f = open('D:\\snake\\classname.csv')
    csv_f = csv.reader(f)
    for row in csv_f:
        if row[1] == top_name:
            classid = row[0]
            sname = row[1]
            ven = row[2]
            cname = row[3]

    view_classify(img_path, probs, classes, top_name, snake_names)

def info():
##    high_classes = top_classes
##    print(high_classes)
##    f = open('G:\\BE proj\\class_info.csv')
##    csv_f = csv.reader(f)
##    for row in csv_f:
##        if row[0] == high_classes:
##            print('found')
##            classid = row[0]
##            sname = row[1]
##            ven = row[2]
##            cname = row[3]


    global a,b,c,d
    a = classid
    b = sname
    c = cname
    d = ven

    s1.set(a)
    s2.set(b)
    s3.set(c)
    s4.set(d)

##    
##def p_and_i():
##    predict()
##    info()
    

root.title("Snake species identification and recognition")
root.geometry("1000x700")
##C=Canvas(root, bg="blue", height=250, width=300)
filename = PhotoImage(file ="brick.gif")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
wel = Label(root, text="WELCOME TO THE APPLICATION", font=("Times New Roman", 25))
wel.place(x=80, y=80, height=75, width=850)
btn = Button(root, text='Open image', command=open_img)
btn.place(x=155, y=180, height=30, width=125)


button=Button(root,text="Predict",command=predict)
button.place(x=590,y=180,height=30,width=125)


info1 = Label(root, text="Classid", image=filename, bd=0, fg="White", compound='center', font=("Comic Sans MS", 14))
info1.place(x=450, y=225, height=30, width=200)

info2 = Label(root, text="Scientific Name", image=filename, bd=0, highlightthickness=0, fg="White", compound='center', font=("Comic Sans MS", 14))
info2.place(x=450, y=275, height=30, width=200)

info3 = Label(root, text="Common Name", image=filename, bd=0, highlightthickness=0, fg="White", compound='center', font=("Comic Sans MS", 14))
info3.place(x=450, y=325, height=30, width=200)

info4 = Label(root, text="Category", image=filename, bd=0, highlightthickness=0, fg="White", compound='center', font=("Comic Sans MS", 14))
info4.place(x=450, y=375, height=30, width=200)

s1=StringVar()
##print(s1)
##st1 = 'Class ID:'+str(s1)
##print(st1)
text1 = Entry(root,textvariable=s1)
text1.place(x=650,y=225,height=25,width=300)

s2=StringVar()
##st2 = 'Scientific Name:'+str(s2)
text2 = Entry(root,textvariable=s2)
text2.place(x=650,y=275,height=25,width=300)

s3=StringVar()
##st3 = 'Common Name:'+str(s3)
text3 = Entry(root,textvariable=s3)
text3.place(x=650,y=325,height=25,width=300)

s4=StringVar()
##st4 = 'Type:'+str(s4)
text4 = Entry(root,textvariable=s4)
text4.place(x=650,y=375,height=25,width=300)


button=Button(root,text="Show Info",command=info)
button.place(x=745,y=180,height=30,width=125)





##s1=StringVar()
##s2=StringVar()
##s3=StringVar()
##s4=StringVar()
##text = Text(root, height=20, width=35)
##text.place(x=550, y=225)
##text.tag_configure('color', foreground='#476042',font=('Tempus Sans ITC', 12, 'bold'))
##st1 = 'Class ID:'+str(s1)
##st2 = 'Scientific Name:'+str(s2)
##st3 = 'Common Name:'+str(s3)
##st4 = 'Type:'+str(s4)
##
##print(st1)
##print(st2)
##print(st3)
##print(st4)
##
##text.insert(END, st1, 'color')
##text.insert(END, st2, 'color')
##text.insert(END, st3, 'color')
##text.insert(END, st4, 'color')





root.mainloop()
