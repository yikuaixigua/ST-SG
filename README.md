# Limited Training Data SAR Image Change Detection via Spatial-Temporal Semantic and Geographic Correlation

The code will be open source after the publication of the paper.
## Test procedure
  ###  1. Basic configuration Settings 
  #### config.yaml
    GPU:
        use: True
        id:   1
    Train:    
        epochs:                15
        batchsize:             32 # batchsize
        finetune_train_ratio:  0.05       # Finetune Training set ratio 
        pretrained_model:     'checkpoints/best_w.pkl' #pre-trained model 
        (*We have saved the pre-trained graph feature representation model with unlabeled training dataset to checkpoints/best_w.pkl)
        finetune:             'finetune_checkpoints/'  #finetune model save path
    Data: # bi-temporal SAR dataset and ground truth
      image1: 'datasets/t1_9.bmp'
      image2: 'datasets/t2_9.bmp'
      label: 'datasets/label9.bmp'
    
### 2. Fine-tune and predict
#### finetune.py
     python finetune.py
     The result is saved as changemap.bmp