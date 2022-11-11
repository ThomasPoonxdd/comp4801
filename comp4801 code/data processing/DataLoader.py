import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import json


class LvisDataset(Dataset):
    def __init__(self,json_path):
        if json_path and os.path.exists(json_path):
            with open(json_path) as user_file:
                file_contents = user_file.read()
            parsed_json = json.loads(file_contents)
            
            # print(parsed_json.keys())
            # dict_keys(['info', 'categories', 'annotations', 'images', 'licenses'])
            # separate them into three different list
            images = parsed_json['images']
            annotations = parsed_json['annotations']
            categories = parsed_json['categories']
            # print(len(images),len(annotations),len(categories))
            new_images = []
            new_annotations = []
            new_categories = []
            images_dic = {}
            categories_dic = {}
            
        
            for i in range(len(images)):
                if images[i]['id'] not in images_dic.keys():
                    images_dic[images[i]['id']] = [images[i]]
                else:
                    images_dic[images[i]['id']].append(images[i])
                    
            for i in range(len(categories)):
                if categories[i]['id'] not in images_dic.keys():
                    categories_dic[categories[i]['id']] = [categories[i]]
                else:
                    categories_dic[categories[i]['id']].append(categories[i])      
            
            for i in range(len(annotations)):
                new_images.append(annotations[i])
                new_annotations.append(images_dic[annotations[i]['image_id']][0])
                new_categories.append(categories_dic[annotations[i]['category_id']][0])
            
            self.images = new_images
            self.annotations = new_annotations
            self.categories =  new_categories
            self.n_samples = len(new_images)  
            
                

    
    def __getitem__(self, index):
        return self.images[index] , self.annotations[index], self.categories[index]

    def __len__(self):
        return self.n_samples
    
if __name__ == '__main__':
    a = LvisDataset(json_path="C:/Users/User/Desktop/lvis_v1_val.json")
    