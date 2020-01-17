## JC DEMO

> run process.py to test unit component
>
> run search_api.py to use component combine with ELK by post json
(call by image path)
>
> run api_server.py to use component by post json
(call by image in type base64)

## Installation
```
pip3 install Pillow opencv-contrib-python matplotlib h5py requests tornado jieba python-Levenshtein shapely hanziconv scikit-learn

pip3 install -v elasticsearch==6.3.1 numpy==1.14.2 scikit-image==0.14.2 scipy==1.2.0
pip3 install -v tensorflow-gpu==1.13.1
pip3 install keras

# mxnet
pip3 install mxnet-cu100

# torch
# pip3 install https://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
# or
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# yolo
pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

# apt-get for cv2
apt-get install libglib2.0-0 libsm6 libxext6 libxrender1

# apt-get for TK
apt-get install python3-tk
```

## pretrained models
download by:
https://drive.google.com/open?id=1ADvzETp45lLQneEzO4Vce9rGDbkR_A5p

contains:
1. obj_dectect_module/resnet50_coco_best_v2.0.1.h5
2. obj_dectect_module/yolo.h5
3. image_vec/pretrained/inception_v3.ckpt
4. face_module/pre_train_models
5. ocr_module/EAST/pretrained_model
6. ocr_module/chinese_ocr/models/*
7. fr_module/model_alignt_person/*
8. fr_module/mtcnn-model/*
-----------------------

## OCR USAGE
```
python demo_chinese.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder /home/user/Pictures/ --recog_model recognition/weights/TPS-ResNet-BiLSTM-Attn.pth --test_folder testpic
```