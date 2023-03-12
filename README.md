Comp4801

1. Dataset (Thomas)
    - LVIS augementation
        - ReizeTransform (default)
        - ExtentTransform  (default)
        - RotationTransform (default)
        - ColorTransform (default)
        - PILColorTransform (default)
        
        - RandomFlip
        - 
    - LVIS data format key,value pair documentation
2. Trainer 
    - DefaultTrainer
3. Predictor
    - DefaultPredictor
4. Visualization ? (Piggy)
    - Tensorboard
    - Demo
5. Video analysis (Piggy)
6. Module Configuration
    - config folder 
        - Base_VILD_RCNN_FPN.yaml
7. Model modules
    - modeling folder

Implementation details: We benchmark on the Mask R-CNN (He et al., 2017) with ResNet (He et al., 2016) FPN (Lin et al., 2017) backbone and use the same settings for all models unless explicitly speciﬁed. The models use 1024×1024 as input image size, large-scale jittering augmentation of range [0.1, 2.0], synchronized batch normalization (Ioffe & Szegedy, 2015; Girshick et al., 2018) of batch size 256, weight decay of 4e-5, and an initial learning rate of 0.32. We train the model from scratch for 180,000 iterations, and divide the learning rate by 10 at 0.9×, 0.95×, and 0.975× of total iterations. We use the publicly available pretrained CLIP model 1 as the open-vocabulary classiﬁcation model, with an input size of 224×224. The temperature τ is set to 0.01, and the maximum number of detections per image is 300. We refer the readers to Appendix D for more details.
