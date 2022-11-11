import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import json
from PIL import Image


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
            # new_images = []
            # new_annotations = []
            # new_categories = []
            images_d = {}
            bbox_d = {}
            
        
            for i in range(len(images)):
                if images[i]['id'] not in images_d.keys():
                    image_name = images[i]['coco_url'].split('/')[-1]
                    path_prefix ="./train/"
                    image_path = os.path.join(path_prefix, image_name)
                    image = Image.open(image_path)
                    to_tensor = torchvision.transforms.ToTensor()
                    image = to_tensor(image)
                    images_d[images[i]['id']] = [image]
                    break
                else:
                    # images_dic[images[i]['id']].append(images[i])
                    print("Error: duplicate image id")
            
            # for i in range(len(categories)):
            #     if categories[i]['id'] not in categories_dic.keys():
            #         categories_dic[categories[i]['id']] = [categories[i]]
            #     else:
            #         categories_dic[categories[i]['id']].append(categories[i])      
            
            for i in range(len(annotations)):
                img_id = annotations[i]['image_id']
                cat_id = annotations[i]['category_id']
                one_hot = torch.zeros(len(categories))
                one_hot[cat_id] = 1
                
                if img_id not in bbox_d.keys():
                    bbox_d[img_id] = one_hot
                else:
                    bbox_d[img_id] = torch.stack((bbox_d[img_id], one_hot), dim = 0)
                break
                # new_images.append(annotations[i])
                # new_annotations.append(images_dic[annotations[i]['image_id']][0])
                # new_categories.append(categories_dic[annotations[i]['category_id']][0])
            
            self.images = images_d
            self.bbox = bbox_d
            # self.annotations = new_annotations
            # self.categories =  new_categories
            # self.n_samples = len(new_images)  
            
                

    
    def __getitem__(self, index):
        return self.images[index] , self.bbox[index]

    def __len__(self):
        return len(self.images)
    
if __name__ == '__main__':
    a = LvisDataset(json_path="C:/Users/User/Desktop/lvis_v1_val.json")
    