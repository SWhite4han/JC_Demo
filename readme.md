## JC DEMO

run process.py to test unit component

run search_api.py to use component combine with ELK by post json
(call by image path)

run api_server.py to use component by post json
(call by image in type base64)

## Installation

pip3 install Pillow scipy opencv-contrib-python matplotlib h5py requests tornado jieba python-Levenshtein shapely hanziconv scikit-learn

pip3 install -v elasticsearch==6.3.1 numpy==1.14.2 scikit-image==0.14.2
pip3 install -v tensorflow-gpu==1.13.1
pip3 install keras

### mxnet
pip3 install mxnet-cu100

--- torch ---
### pip3 install https://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision

--- yolo ---
pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

--- apt-get ---
### for cv2
apt-get install libglib2.0-0 libsm6 libxext6 libxrender1

### for TK
apt-get install python3-tk

------- models --------
https://drive.google.com/open?id=1ADvzETp45lLQneEzO4Vce9rGDbkR_A5p

contains:
obj_dectect_module/resnet50_coco_best_v2.0.1.h5
obj_dectect_module/yolo.h5
image_vec/pretrained/inception_v3.ckpt
face_module/pre_train_models
ocr_module/EAST/pretrained_model
ocr_module/chinese_ocr/models/*
-----------------------