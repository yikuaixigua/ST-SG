# Limited Training Data SAR Image Change Detection via Spatial-Temporal Semantic and Geographic Correlation

The code will be open source after the publication of the paper.
## Test procedure
  ###  1. Basic configuration Settings 
please download the pre-trained model at 
[**pre-trained model**](https://drive.google.com/file/d/1H-SrJZHFNBwFjwTMViVtEUznPc8bsDkH/view?usp=sharing)
, which we have pre-trained with unlabeled datasets.
  #### config.yaml
    GPU:
        use: True
        id:   1
    Train:    
        epochs:                30
        batchsize:             32 # batchsize
        finetune_train_ratio:  0.05       # Finetune Training set ratio 
        pretrained_model:     'checkpoints/best_w.pkl' # The downloaded pre-trained model is placed in this folder 
        finetune:             'finetune_checkpoints/'  # finetune model save path
    Data: # bi-temporal SAR dataset and ground truth
      image1: 'datasets/t1_9.bmp'
      image2: 'datasets/t2_9.bmp'
      label: 'datasets/label9.bmp'
    
### 2. Fine-tune and predict
#### finetune.py
     python finetune.py
     The result is saved as changemap.bmp# ST-SGNet

-------------------
## Complex-valued version
### 1. The amplitude and phase features are obtained from the real and imaginary parts
#### brpi.py
    
###  2. Basic configuration Settings 
  #### config.yaml
    Data: # bi-temporal SAR complex-valued dataset and ground truth
      image1_i: 'datasets/t1_9.bmp' # real part
      image2_i: 'datasets/t2_9.bmp'
      image1_q: 'datasets/t1_9.bmp' # imaginary part
      image2_q: 'datasets/t2_9.bmp'
      label: 'datasets/label9.bmp'
### 3. Fine-tune and predict