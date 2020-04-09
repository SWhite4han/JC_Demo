"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import string

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils

from craft import CRAFT
from recognition.model import Model
from recognition.utils import CTCLabelConverter, AttnLabelConverter
from detection import Detection
from recognition.recognition import Recognition
from collections import OrderedDict

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
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
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
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--recog_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parser()
    """ vocab / character number configuration """
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)

    result_folder = './result/det_recog/'  # recognition
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    """ load detection and recognition net """
    net_detection = Detection(args)     # initialize
    net_recog = Recognition(args)


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

        img = image[:,:,::-1]
        img = imgproc.normalizeMeanVariance(img)

        dst_pts = np.array([[0, 0], [args.imgW, 0], [args.imgW, args.imgH], [0, args.imgH]], dtype=np.float32)

        detected_text = torch.Tensor()

        ## transform detected box to input size of recognition
        for i, box in enumerate(polys):
            M = cv2.getPerspectiveTransform(box, dst_pts)
            warp = cv2.warpPerspective(img, M, (args.imgW, args.imgH))
            warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            warp = warp[:, :, np.newaxis]  # [h, w] to [h, w, 1]
            x = torch.from_numpy(warp).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
            x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
            detected_text = torch.cat((detected_text, x), dim=0)
        if args.cuda:
            detected_text = detected_text.cuda()

        preds_str = net_recog.predict(detected_text)

        if 'Attn' in args.Prediction:
            for idx, pred in enumerate(preds_str):
                preds_str[idx] = pred[:pred.find('[s]')]

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder,  texts=preds_str)

    print("elapsed time : {}s".format(time.time() - t))
