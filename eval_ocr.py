import os
import cv2
import numpy as np
from scipy import spatial
import Levenshtein

from text_recognition.demo_chinese import parser, inference
from text_recognition.detection import Detection
from text_recognition.recognition.recognition import Recognition, demo
from text_recognition.ocr_module.chinese_ocr.eval import OCR
from text_recognition import imgproc
from Common.common_lib import get_images

import xml.etree.cElementTree as ET

""" load detection and recognition net """
arguments = parser()
net_detection = Detection(arguments)  # initialize
chinese_recog = OCR()  # Chinese OCR

base_path = "/home/user/Downloads/icdar2003/test/"

# Create acc_dict for storing targets and results.
acc_dict = dict()
tree = ET.ElementTree(file=os.path.join(base_path, "word.xml"))
root = tree.getroot()
for child in root:
    acc_dict[child.attrib.get("file")] = {'tag': child.get('tag')}

# For storing evaluate results
total = 0
corrects = 0
distances = 0
ratios_list = list()


# # Load images.
# image_list = get_images(os.path.join(base_path, "word"))
#
# # Hand made evaluation
# Evaluate by each image.
# for path in image_list:
#     try:
#         rlts = inference(net_detection, chinese_recog, [path])
#         if rlts:
#             # print(acc_dict[path.replace(base_path, '')]['tag'], rlts[0])
#             total += 1
#             target = acc_dict[path.replace(base_path, '')]['tag'].lower()
#             result = rlts[0].lower()
#             if target == result:
#                 corrects += 1
#
#             distances += Levenshtein.distance(target, result)
#             ratio = Levenshtein.ratio(target, result)
#             ratios_list.append(ratio)
#         else:
#             pass
#             # print(acc_dict[path.replace(base_path, '')]['tag'], path)
#     except Exception as e:
#         # print(path)
#         print(e)

# Evaluate by call func
results, image_path = demo(arguments)
for i, path in enumerate(image_path):
    total += 1
    target = acc_dict[path.replace(base_path, '')]['tag'].lower()
    result = results[i].lower()

    if target == result:
        corrects += 1

    distances += Levenshtein.distance(target, result)
    ratio = Levenshtein.ratio(target, result)
    ratios_list.append(ratio)
    print('{}, {}, {}'.format(target, result, ratio))

# print result
ratios = np.mean(np.asarray(ratios_list))

# print("accuracy: %.4f" % (corrects / total))
print("mean similarity: %.4f" % ratios)

