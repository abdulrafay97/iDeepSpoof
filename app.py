#Import Packages
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import models, transforms
import torch.nn as nn
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

icon = Image.open('icon.png')
st.set_page_config(page_title='iDeepSpoof', page_icon = icon)
st.header('iDeepSpoof')
st.sidebar.subheader(("Input a picture of a Face."))

#Load Model
def resnet():
    model = models.resnet50(pretrained=False).to(device)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 11)).to(device)
    model.load_state_dict(torch.load('weights.h5', map_location=torch.device(device)))
    model.eval()

    return model

#Calculating Prediction
def Predict(img):
    allClasses = ['3D Mask', 'A4', 'Face Mask', 'Live',
		'Pad', 'PC', 'Phone', 'Photo', 'Poster',
		'Region Mask', 'Upper Body Mask']
    Mod = resnet()
    out = Mod(img)
    _, predicted = torch.max(out.data, 1)
    allClasses.sort()
    labelPred = allClasses[predicted]
    return labelPred



#Get Image
file_up = st.sidebar.file_uploader('Upload an Image', type = "jpg")

#Normalizing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#Transforming the Image
data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up).convert('RGB')
    st.image(image, caption = 'Uploaded Image.', width=None, use_column_width=None)
    st.write("")
    st.write("Just a second ...")
    img = data_transform(image)
    img = torch.reshape(img , (1, 3, 224, 224))
    prob = Predict(img)
    st.write(prob)
