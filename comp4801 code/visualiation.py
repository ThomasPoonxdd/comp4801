import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2


def visual(inform,img):
    
    # for each boundary box 
    for i in range(len(inform['boxes'].item()[0])):
        # draw boundary box 
        cv2.rectangle(img,(inform['boxes'].item()[0][i][0],inform['boxes'].item()[0][i][1]),(inform['boxes'].item()[0][i][2],inform['boxes'].item()[0][i][3]), (255,0,0), 2)
        # get the text size
        (w, h), _ = cv2.getTextSize(f"{inform['labels'].item()[i]}:{inform['scores'].item()[i]}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Prints the text.    
        # img = cv2.rectangle(img, (boundary_box_co[0],boundary_box_co[1]-20),(boundary_box_co[0]+w,boundary_box_co[1]), (255,0,0), -1)
        img = cv2.putText(img, f"{inform['labels'].item()[i]}:{inform['scores'].item()[i]}", (inform['boxes'].item()[0][i][0],inform['boxes'].item()[0][i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 1)
        # For printing text
        # img = cv2.putText(img, 'test', (boundary_box_co[0],boundary_box_co[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    plt.imshow(img)
    plt.show()

# [objectness, bounding box] = [tensor(B,N_box),Tensor(B,N_box,4)]

if __name__ == "__main__":
    #array = [{'boxes': torch.tensor()}]
    #path = "C:/Users/user/Desktop/a.jpeg"
    #img = cv2.imread(path)
    #print(array[0][1][1].item())
    #visual(array,img)