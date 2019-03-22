# tornado server
import sys
import os
import time
import operator
import json
import requests
import tensorflow as tf
import tornado.ioloop
import tornado.options
import tornado.web
import cv2
import logging
from tensorflow.python.tools import inspect_checkpoint as chkp
from logging.config import fileConfig

from nlp_module.ner.eval import ner_obj
from face_module.facenet import facenet_obj
from face_module.face2vec import face2vec_for_query
from obj_dectect_module.Detection import Detection
from image_vec.img2vec import imagenet_obj
from ocr_module.chinese_ocr.eval import OCD, OCR
from PIL import Image
from Common.common_lib import string2img, img2string


def submit(url, json=None):
    resp = requests.post(url=url, json=json)
    return resp


class MainHandler(tornado.web.RequestHandler):

    def initialize(self, args_dict):
        # self.OCD = args_dict['OCD']
        # self.OCR = args_dict['OCR']
        # self.yolo = args_dict['yolo']
        # self.facenet = args_dict['facenet']
        self.imagenet = args_dict['imagenet']
        # self.ner = args_dict['ner']

    def post(self):
        """
        task: 0: yolo + facenet
              1: imagenet
              2: ocd + ocr
              3: ner
        :return:
        """
        # 接收資料
        post_data = self.request.body.decode('utf-8')
        post_data = json.loads(post_data)
        task = post_data['task']
        image = string2img(post_data['image'])

        # image = cv2.imread(r'/data1/images/川普/google/000004.jpg')

        if task == '0':
            image = [image]
            ob = face2vec_for_query(self.yolo, self.facenet, image)
            self.write(json.dumps(ob))

        elif task == '1':
            image = [image]
            _, v = self.imagenet.new_img_vec(image)
            # vec = [float(value) for value in v]
            self.write(json.dumps({'imgVec': v[0].tolist(), 'response': 'good'}))

        elif task == '2':
            # EAST
            image_list, masked_image, boxes = self.OCD.detection(image)

            b64 = img2string(masked_image)

            # cv2 to pil
            cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            img = pil_im.convert("RGB")

            # ocr
            result_list = self.OCR.recognize(img, boxes)
            text_list = dict()
            for idx, elem in enumerate(result_list):
                text_list[idx] = elem['text']

            # --
            # # cv2 to pil, and show
            # cv2_im = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            # pil_im = Image.fromarray(cv2_im)
            # pil_im.show()
            # for idx, elem in enumerate(result_list):
            #     print("text: %s" % elem['text'])
            #
            #     # cv2 to pil
            #     cv2_im = cv2.cvtColor(image_list[idx], cv2.COLOR_BGR2RGB)
            #     pil_im = Image.fromarray(cv2_im)
            #     pil_im.show()
            # --

            self.write({'result_text': text_list, 'result_image': b64})

        elif task == '3':
            pass

        # self.write("ok")


def make_app():
    # ocd = OCD("ocr_module/EAST/pretrained_model/east_mixed_149482/")
    # ocr = OCR()
    # ner = ner_obj()
    # yolo = Detection()
    # facenet = facenet_obj()
    imagenet = imagenet_obj()

    args_dict = {
        # "OCD": ocd,
        # "OCR": ocr,
        # "ner": ner,
        # "yolo": yolo,
        # "facenet": facenet,
        "imagenet": imagenet,
    }
    application = tornado.web.Application([(r"/", MainHandler, dict(args_dict=args_dict))])
    # http_server = tornado.httpserver.HTTPServer(application, max_buffer_size=100000)  # default = 100M
    http_server = tornado.httpserver.HTTPServer(application)
    return http_server


def main(configure):

    # service
    serv = make_app()

    # 設定 port 為 8080
    serv.listen(9527)

    print("server start")

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main(sys.argv[0])
