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
from nlp_module import cc
from face_module.facenet import facenet_obj
from face_module.face2vec import face2vec_for_sol_data
from obj_dectect_module.Detection import Detection
from image_vec.img2vec import imagenet_obj
from ocr_module.chinese_ocr.eval import OCD, OCR
from PIL import Image
from Common.common_lib import download_image
# import upload func
from TestElasticSearch import NcsistSearchApiPath as InfinitySearchApi
from configuration.config import Config


class MainHandler(tornado.web.RequestHandler):

    def initialize(self, args_dict):
        self.OCD = args_dict['OCD']
        self.OCR = args_dict['OCR']
        self.yolo = args_dict['yolo']
        self.facenet = args_dict['facenet']
        self.imagenet = args_dict['imagenet']
        # self.ner = args_dict['ner']
        self.es = args_dict['es_obj']
        self.index = args_dict['index']
        self.config = args_dict['config']
        self.checklist = args_dict['url_checklist']

    def get_images_feature(self, image_paths):
        """

        :param image_paths: a list of image paths
        :return:
        """
        images = []
        len_imgs = len(image_paths)
        if len_imgs < 1:
            return None
        for path in image_paths:
            # Modify for SOL DATA
            images.append(cv2.imread(download_image(path)))

        _, vectors = self.imagenet.new_img_vec(images)
        return vectors

    def get_face_features(self, image_urls):
        """

        :param image_urls: a list of image urls
        :return:
        """
        images = []
        image_ps = []
        len_imgs = len(image_urls)
        if len_imgs < 1:
            return None
        for url in image_urls:
            path = download_image(url)
            image_ps.append(path)
            images.append(cv2.imread(path))
        face_vectors, face_source, locations = face2vec_for_sol_data(self.yolo, self.facenet, images, image_ps)
        # exist_paths = [image_urls[i] for i in indices]
        return face_vectors, face_source, locations

    def get_ocr_result(self, image_path):
        len_imgs = len(image_path)

        # Only calculate first image
        img = image_path[0]

        img = download_image(img)

        image = cv2.imread(img)

        # EAST
        image_list, masked_image, boxes = self.OCD.detection(image)

        if image_list:
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

            return text_list
        else:
            return []

    def check_redundant(self, paths):
        new_list = list()
        status = dict()
        for p in paths:
            if p not in self.checklist:
                new_list.append(p)
                status[p] = {'status': 'wait'}
            else:
                status[p] = {'status': 'redundant'}
        return new_list, status

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
        paths = post_data['img_paths']
        top_k = post_data.get('top')
        score_threshold = post_data.get('threshold')
        if not top_k:
            top_k = 10
        if not score_threshold:
            score_threshold = 0.5
        print(post_data)

        if task == '0':
            # image = [image]
            face_vectors, exist_paths, locations = self.get_face_features(paths)

            if face_vectors:
                ob = {'imgVec': face_vectors[0].tolist(), 'response': 'good'}
                # ob2 = {'imgVec': [f.tolist() for f in face_vectors], 'response': 'good'}
                if self.es:
                    result = self.es.query_face_result(face_vectors[0].tolist(), target_index=self.index, top=top_k)
                    score_result = [each_rlt for each_rlt in result if each_rlt['_score'] > score_threshold]
                    self.write(json.dumps(score_result))

        elif task == '1':
            v = self.get_images_feature(paths)
            # ################################################### search img with v[0].tolist() #####################################
            if v:
                ob = {'imgVec': v[0].tolist(), 'response': 'good'}
                if self.es:
                    result = self.es.query_image_result(v[0].tolist(), target_index=self.index, top=top_k)
                    score_result = [each_rlt for each_rlt in result if each_rlt['_score'] > score_threshold]
                    self.write(json.dumps(score_result))

        elif task == '2':

            text_list = self.get_ocr_result(paths)

            # self.write({'result_text': text_list, 'result_image': b64})
            if text_list:
                for i, v in text_list.items():
                    text_list[i] = cc.trans_s2t(v)
                self.write({'result_text': text_list})
            else:
                self.write({'result_text': {}})

        elif task == '3':
            pass
        elif task == 'upload_img':

            # wait_list, image_status = self.check_redundant(paths=paths)

            # ---------------------------------
            # Upload Face
            # ---------------------------------
            face_vectors, exist_paths, _ = self.get_face_features(paths)
            if face_vectors:
                if len(face_vectors) > 0:
                    tmp = ''
                    count_same_img = 1
                    for i in range(len(face_vectors)):
                        source_path = exist_paths[i].split('/')[-1]  # only save pic name
                        if tmp == exist_paths[i]:
                            count_same_img += 1
                        else:
                            tmp = exist_paths[i]
                            count_same_img = 1

                        self.es.push_data({'imgVec': face_vectors[i].tolist(),
                                           'category': 'face',
                                           'imgPath': source_path}, target_index=self.index)
                    print('face ok.')
            else:
                print('no face')
            # ---------------------------------
            # Upload Image Feature
            # ---------------------------------
            vecs = self.get_images_feature(image_paths=paths)
            if len(vecs) > 0:
                for i in range(len(vecs)):
                    source_path = paths[i].split('/')[-1]
                    vec = vecs[i].tolist()
                    if len(vec) == 2048:
                        self.es.push_data({'imgVec': vec,
                                           'category': 'img',
                                           'imgPath': source_path}, target_index=self.index)
                print('image ok.')
            # ---------------------------------
            self.write({"response": "Upload succeeded."})
            pass
        # self.write("ok")


def cmd_connect_es():
    try:
        es_obj = InfinitySearchApi.InfinitySearch('hosts=10.10.53.201,10.10.53.204,10.10.53.207;port=9200;id=esadmin;passwd=esadmin@2018')
        status = es_obj.status()
        print(status)
        return es_obj
    except Exception as ex:
        print(ex)


def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            check_file = json.load(f)
    else:
        check_file = {}
    return check_file


def dump_json_file(file_path, target):
    with open(file_path) as f:
        json.dump(target, f)


def make_app():
    cfg = Config()
    ocd = OCD("ocr_module/EAST/pretrained_model/east_mixed_149482/")
    ocr = OCR()
    # ner = ner_obj()
    yolo = Detection()
    facenet = facenet_obj()
    imagenet = imagenet_obj()

    es = cmd_connect_es()

    url_checklist = load_json_file(cfg.url_checklist_path)

    args_dict = {
        "OCD": ocd,
        "OCR": ocr,
        # "ner": ner,
        "yolo": yolo,
        "facenet": facenet,
        "imagenet": imagenet,
        "es_obj": es,
        "config": cfg,
        "url_checklist": url_checklist,
        # "index": "ncsist_test",
        "index": "ui_test",  # ****************************
    }
    application = tornado.web.Application([(r"/", MainHandler, dict(args_dict=args_dict))])
    # http_server = tornado.httpserver.HTTPServer(application, max_buffer_size=100000)  # default = 100M
    http_server = tornado.httpserver.HTTPServer(application)
    return http_server


def main(configure):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    # service
    serv = make_app()

    # 設定 port 為 8080
    serv.listen(9528)

    print("server start")

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main(sys.argv[0])
