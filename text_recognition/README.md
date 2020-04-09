This work is inspired by following github.

## CRAFT: Character-Region Awareness For Text detection
Official Pytorch implementation of CRAFT text detector | github(https://github.com/clovaai/CRAFT-pytorch) | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

## Text recognition
Official Pytorch implementation of text recognizer | github(https://github.com/clovaai/deep-text-recognition-benchmark) | [Paper](https://arxiv.org/pdf/1904.01906) | [Pretrained Model](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)
 

## Getting started
### Install dependencies
#### Requirements
- PyTorch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2
- check requiremtns.txt
```
pip install -r requirements.txt
```


### Test instruction using pretrained model

Recognition Test (defaults weights path : recognition/weights; output path: result/recognition)
```
cd recognition
python3 recog_demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder [folder path to test images] --recog_model weights/TPS-ResNet-BiLSTM-Attn.pth
```
Detection Test (defaults weights path : recognition/weights; output path: result/detection)
```
python3 test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```
Detection and Recognition Test (defaults weights path : weights/craft_mlt_25k.pth; output path: result/det_recog)
```
python3 demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --test_folder demo_image/ --recog_model recognition/weights/TPS-ResNet-BiLSTM-Attn.pth 
```
### Evaluate
#### load predict and gt xml 
* gt : https://drive.google.com/file/d/1yFn1jkSEYYbq-1sYSrpWFI4GxBC6QunV/view?usp=sharing
* predict : https://drive.google.com/file/d/1vUKBQ2FCSGipjv0kXMlir6KTlUzQJdif/view?usp=sharing

#### evaluate
```
python3 evaluate.py
```


### Arguments
* `--trained_model`: pretrained model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--test_folder`: folder path to input images

## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}

@inproceedings{baek2019STRcomparisons,
  title={What is wrong with scene text recognition model comparisons? dataset and model analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  note={to appear},
  pubstate={published},
  tppubtype={inproceedings}
}
```
