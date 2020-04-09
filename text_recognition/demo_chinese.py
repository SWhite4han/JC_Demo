"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse
import string

import torch

from PIL import Image

import cv2
import numpy as np
from text_recognition import imgproc
from text_recognition import file_utils

from text_recognition.detection import Detection
from text_recognition.recognition.recognition import Recognition
from collections import OrderedDict
from text_recognition.ocr_module.chinese_ocr.eval import OCR

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def parser():
    parser = argparse.ArgumentParser(description='Text Detection and Recognition')

    """Detection"""
    parser.add_argument('--trained_model', default='text_recognition/weights/craft_mlt_25k.pth',
                        type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.6, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='data/SceneTrialTest/', type=str, help='folder path to input images')

    """Recognition"""
    parser.add_argument('--image_folder', default="/home/user/Downloads/icdar2003/test/word/",
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--recog_model', default="text_recognition/recognition/weights/TPS-ResNet-BiLSTM-Attn.pth",
                        help="path to saved_model to evaluation")

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')

    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    arguments = parser.parse_args()
    return arguments


def inference(detector, recognizer, image_list):
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)
        # Detection
        bboxes, polys, score_text = detector.predict(image)

        cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        img = pil_im.convert("RGB")

        # Recognition
        result_list = recognizer.recognize(img, bboxes)
        preds_str = list()
        for idx, pred in enumerate(result_list):
            preds_str.append(pred.get('text'))
        return preds_str


if __name__ == '__main__':
    args = parser()
    """ vocab / character number configuration """
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)

    result_folder = './result/det_recog/'  # recognition
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    """ load detection and recognition net """
    net_detection = Detection(args)     # initialize
    chinese_recog = OCR()   # Chinese OCR


    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = net_detection.predict( image)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        # -------------- START: OCR --------------
        cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        img = pil_im.convert("RGB")

        # ocr
        result_list = chinese_recog.recognize(img, bboxes)
        preds_str = list()
        for idx, pred in enumerate(result_list):
            preds_str.append(pred.get('text'))
        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder, texts=preds_str)

        # -------------- END: OCR --------------

    print("elapsed time : {}s".format(time.time() - t))


