# Limited Training Data SAR Image Change Detection via Spatial-Temporal Semantic and Geographic Correlation

The full code will be open source after the publication of the paper. If you have any questions, please contact at lhl_hit@hotmail.com
# Unsupervied Condition
## For Bi-Temporal SAR intensity image Change Detection 
  ###  1. Basic configuration Settings 
  #### config.yaml
    GPU:
        use: True
        id:   1
    Train:    
        epochs:                30
        batchsize:             32 # batchsize
    Data: # bi-temporal SAR dataset and ground truth
      image1: 'datasets/t1_9.bmp'
      image2: 'datasets/t2_9.bmp'
      label: 'datasets/label9.bmp'
    
### 2. Pre-Training and Predict
#### train.py
     python train.py
     The result is saved as changemap.bmp  

# Limited supervied Condition
###  1. Basic configuration Settings 
  #### config.yaml
    GPU:
        use: True
        id:   1
    Train:    
        epochs:                30
        batchsize:             32 # batchsize
        finetune_train_ratio:  0.05       # Finetune Training set ratio 
        pretrained_model:     'checkpoints/best_w.pkl' # The pre-trained model is placed in this folder for fine-tune
        finetune:             'finetune_checkpoints/'  # finetune model save path
    Data: # bi-temporal SAR dataset and ground truth
      image1: 'datasets/t1_9.bmp'
      image2: 'datasets/t2_9.bmp'
      label: 'datasets/label9.bmp'
    
### 2. Fine-tuning and Predict
#### train.py
     python finetune.py
     The result is saved as changemap.bmp  

-------------------
## For Bi-Temporal SAR Complex-valued image Change Detection
    
###  1. Basic configuration Settings 
  #### config.yaml
    Complex_data: # bi-temporal SAR complex-valued dataset and ground truth
      image1: 'datasets/1_2.bmp'
      image2: 'datasets/2_2.bmp'
      image1_i: 'datasets/i1.bin' # real part
      image2_i: 'datasets/i2.bin'
      image1_q: 'datasets/q1.bin' # imaginary part
      image2_q: 'datasets/q2.bin'
      label: 'datasets/label12.bmp'
      img_height: 600
      img_width: 600

### 3. Pre-Training and predict
#### train_complex.py
      python train_complex.py
     The result is saved as changemap.bmp 
