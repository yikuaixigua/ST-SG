# Limited Training Data SAR Image Change Detection via Spatial-Temporal Semantic and Geographic Correlation

The code will be open source after the publication of the paper. If you have any questions, please contact lhl_hit@hotmail.com
# How to use ?
## For SAR intensity image 
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
    
### 2. Fine-tune and Predict
#### finetune1.py
     python finetune1.py
     The result is saved as changemap.bmp  

-------------------
## For Complex-valued SAR
    
###  1. Basic configuration Settings 
  #### config.yaml
    Data: # bi-temporal SAR complex-valued dataset and ground truth
      image1: 'datasets/1_2.bmp'
      image2: 'datasets/2_2.bmp'
      image1_i: 'datasets/i1.bin' # real part
      image2_i: 'datasets/i2.bin'
      image1_q: 'datasets/q1.bin' # imaginary part
      image2_q: 'datasets/q2.bin'
      label: 'datasets/label12.bmp'
      img_height: 600
      img_width: 600

### 3. Fine-tune and predict
#### finetune_complex.py
      python finetune_complex.py
     The result is saved as changemap.bmp 
