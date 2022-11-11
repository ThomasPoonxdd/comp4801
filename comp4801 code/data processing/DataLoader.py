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
            
            images = parsed_json['images']
            annotations = parsed_json['annotations']
            print(len(images),len(annotations))
            dic = {}
            for i in range(len(annotations)):
                if annotations[i]['image_id'] not in dic.keys():
                    dic[annotations[i]['image_id']] = [annotations[i]]
                else:
                    dic[annotations[i]['image_id']].append(annotations[i])
            print(len(dic),dic)
            
                

    
    # def __getitem__(self, index):
    #     return self.x[index] , self.y[index]

    # def __len__(self):
    #     return self.n_samples
    
if __name__ == '__main__':
    a = LvisDataset(json_path="C:/Users/User/Desktop/lvis_v1_val.json")
    