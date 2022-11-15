import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
import json

class dataset(data.Dataset):
    
    # initialise function of class
    # images: location of the images dictionary
    # info: location of the json file with images' info
    def __init__(self, images, info):

        # list of names of images
        filenames = []

        # list of category ids
        labels = []

        # retrieve information from json
        f = open(info)
        data = json.load(f)
        for i in data['annotations']:
            fname = str(i['id']).zfill(12) + '.jpg'
            filenames.append(fname)
            labels.append(i['category_id'])
        f.close()

        # the data directory 
        self.images = images
        # the list of filename
        self.filenames = filenames
        # the list of label
        self.labels = labels

    # obtain the sample with the given index
    def __getitem__(self, index):
        # obtain filenames from list
        image_filename = self.filenames[index]
        # Load data and label
        image = Image.open(os.path.join(self.images, image_filename))
        label = self.labels[index]
        
        # output of Dataset must be tensor
        image = transforms.ToTensor()(image)
        label = torch.as_tensor(label, dtype=torch.int64)
        return image, label
    
    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)
