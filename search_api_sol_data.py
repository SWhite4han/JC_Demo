# tornado server
import sys
import os
import json
import tornado.ioloop
import tornado.options
import tornado.web
import cv2
import numpy as np
import logging
from tensorflow.python.tools import inspect_checkpoint as chkp
from logging.config import fileConfig

# from nlp_module.ner.eval import ner_obj
# from nlp_module import cc
from image_vec.img2vec import imagenet_obj
from PIL import Image
from Common.common_lib import download_image, img2string, string2img
from TestElasticSearch import NcsistSearchApiPath as InfinitySearchApi
from configuration.config import Config
from fr_module import face_model
from text_recognition.demo_chinese import parser, inference
from text_recognition.detection import Detection
from text_recognition.ocr_module.chinese_ocr.eval import OCR


class MainHandler(tornado.web.RequestHandler):

    def initialize(self, args_dict):
        self.OCD = args_dict['OCD']
        self.OCR = args_dict['OCR']
        self.imagenet = args_dict['imagenet']
        self.es = args_dict['es_obj']
        self.index = args_dict['index']
        self.config = args_dict['config']
        self.checklist = args_dict['url_checklist']
        self.arc_face = args_dict['arcface']
        self.log = args_dict['logger']
        self.store_path = args_dict['store_path']

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
            # Modify for SOL DATA (use file path instead of image)
            try:
                images.append(cv2.imread(download_image(path, self.store_path)))
            except Exception as e:
                self.log.error(e)

        imgs, vectors = self.imagenet.new_img_vec(images)
        return vectors

    def get_face_location_arcface(self, image_url):
        """

        :param image_paths: a list of image paths
        :return:
        """
        show_faces = []

        path = download_image(image_url, self.store_path)
        raw_image = cv2.imread(path)
        aligned_imgs, bound_boxs = self.arc_face.get_multi_input(raw_image)
        if aligned_imgs is not None:
            for i in range(len(aligned_imgs)):
                # aligned_img = aligned_imgs[i]
                bbox = bound_boxs[i]
                # face = cv2.cvtColor(raw_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], cv2.COLOR_BGR2RGB)
                face = raw_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                show_faces.append(img2string(face))
            return aligned_imgs, bound_boxs, show_faces

    def get_face_feature_arcface(self, vec):
        return self.arc_face.get_feature(vec)

    def get_ocr_result(self, image_path):

        # Only calculate first image
        img = image_path[0]

        try:
            img = download_image(img, self.store_path)
            # CRAFT
            rlts = inference(self.OCD, self.OCR, [img])

            if rlts:
                results = dict()
                for i, v in enumerate(rlts):
                    results[str(i)] = v
                return results
            else:
                return []
        except Exception as e:
            self.log.error(e)

    def check_redundant(self, paths):
        new_list = list()
        status = dict()
        for p in paths:
            if p not in self.checklist:
                new_list.append(p)
            else:
                status[p] = 'redundant'
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
        paths = post_data.get('img_paths')
        top_k = post_data.get('top')
        score_threshold = post_data.get('threshold')
        feature_vec = post_data.get('feature')
        if not top_k:
            top_k = 10
        if not score_threshold:
            score_threshold = 0.5
        self.log.info(post_data)

        if task == '0':
            pass
        elif task == '1':
            v = self.get_images_feature(paths)
            # ################################################### search img with v[0].tolist() #####################################
            if v:
                # ob = {'imgVec': v[0].tolist(), 'response': 'good'}
                if self.es:
                    result = self.es.query_image_result(v[0].tolist(), target_index=self.index, top=top_k)
                    score_result = [each_rlt for each_rlt in result if each_rlt['_score'] > score_threshold]
                    self.log.info('return:{0}'.format(score_result))
                    self.log.info('len of return:{0}'.format(len(score_result)))
                    self.write(json.dumps(score_result))

        elif task == '2':

            text_list = self.get_ocr_result(paths)

            # self.write({'result_text': text_list, 'result_image': b64})
            if text_list:
                # force transport simple chinese to traditional chinese
                # for i, v in text_list.items():
                #     text_list[i] = cc.trans_s2t(v)
                self.write({'result_text': text_list})
            else:
                self.write({'result_text': {}})

        elif task == '3':
            pass
        elif task == '4':
            img_url = paths[0]
            try:
                aligned_imgs, bound_boxs, show_faces = self.get_face_location_arcface(img_url)
                if show_faces:
                    num_face = len(show_faces)
                    message = {'total': num_face, 'face': []}
                    for i in range(num_face):
                        message['face'].append({'pic': show_faces[i], 'feature': aligned_imgs[i].tolist()})
                    self.write(message)
                else:
                    self.write({'total': 0, 'face': []})
            except Exception as e:
                self.log.error(e)
                self.write({'total': 0, 'face': []})

        elif task == '5':
            face_vectors = self.get_face_feature_arcface(np.array(feature_vec))
            if self.es:
                result = self.es.query_face_result(
                    face_vectors.tolist(),
                    target_index=self.index,
                    top=top_k,
                    method='im_cosine'
                )
                score_result = [each_rlt for each_rlt in result if each_rlt['_score'] > score_threshold]
                self.write(json.dumps(score_result))
        elif task == 'upload_img':
            wait_list, redundants = self.check_redundant(paths=paths)  #####################################################
            paths = wait_list
            upload_check = dict()
            upload_check.update(redundants)
            # ---------------------------------
            # Upload Face
            # ---------------------------------
            for path in paths:
                try:
                    aligned_imgs, bound_boxs, show_faces = self.get_face_location_arcface(path)
                except Exception as e:
                    self.log.warning(e)
                    aligned_imgs = None  # For check
                if aligned_imgs is not None:
                    for aligned_img in aligned_imgs:
                        face_vectors = self.get_face_feature_arcface(aligned_img)
                        source_path = path.split('/')[-1]
                        self.es.push_data({'imgVec': face_vectors.tolist(),
                                           'category': 'face',
                                           'imgPath': source_path}, target_index=self.index)
                    self.log.info('{0} face ok.'.format(path))
                    upload_check[path] = 'ok'
                else:
                    self.log.info('{0} no face'.format(path))
                    upload_check[path] = 'fail'
            # ---------------------------------
            # Upload Image Feature
            # ---------------------------------
            if paths:
                for i in range(len(paths)):
                    path = paths[i]
                    try:
                        vecs = self.get_images_feature(image_paths=[path])
                        if len(vecs) > 0 and len(vecs[0]) == 2048:
                            source_path = paths[i].split('/')[-1]
                            vec = vecs[0].tolist()
                            self.es.push_data({'imgVec': vec,
                                               'category': 'img',
                                               'imgPath': source_path}, target_index=self.index)
                            # if upload_check.get(path) != 'fail':
                            upload_check[path] = 'ok'
                        else:
                            upload_check[path] = 'fail'
                    except Exception as e:
                        upload_check[path] = 'fail'
                        self.log.warning(e)

                self.log.info('image ok.')
                # ---------------------------------
            # ---------------------------------
            # OCR Result
            # ---------------------------------
            # test_rlt = {
            #     '0': 'This',
            #     '1': 'is',
            #     '2': 'test',
            #     '3': 'example',
            # }
            # ocr_result = dict()
            # for path in paths:
            #
            #     if upload_check.get(path) == 'ok':
            #         ocr_result[path] = test_rlt
            #     else:
            #         ocr_result[path] = {}

            ocr_result = dict()
            paths.extend(list(redundants.keys()))
            for path in paths:
                text_list = self.get_ocr_result([path])

                if text_list:
                    ocr_result[path] = text_list
                else:
                    ocr_result[path] = {}

            # Saving check list.
            self.checklist.update(upload_check)
            dump_json_file(os.path.join(self.config.url_checklist_path, '{0}_checklist.json'.format(self.config.index)),
                           self.checklist)
            self.log.info('Images are uploaded.')
            self.write({"upload_check": upload_check, "ocr_result": ocr_result})


def cmd_connect_es():
    try:
        es_obj = InfinitySearchApi.InfinitySearch('hosts=182.0.0.71,182.0.0.73,182.0.0.75;port=9200;id=esadmin;passwd=esadmin@2018')
        status = es_obj.status()
        print(status)
        return es_obj
    except Exception as ex:
        print(ex)


def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            check_file = json.load(f)
    else:
        check_file = {}
        dump_json_file(file_path, check_file)
    return check_file


def dump_json_file(file_path, target):
    with open(file_path, 'w') as f:
        json.dump(target, f, ensure_ascii=False)


def log():
    # 基礎設定
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler('api_server.log', 'a', 'utf-8'), ])

    # 定義 handler 輸出 sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 設定輸出格式
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # handler 設定輸出格式
    console.setFormatter(formatter)
    # 加入 hander 到 root logger
    logging.getLogger('').addHandler(console)

    log = logging.getLogger(__name__)
    return log


def make_app():
    cfg = Config()
    arguments = parser()
    # ocd = OCD("ocr_module/EAST/pretrained_model/east_mixed_149482/")
    # ocr = OCR()
    ocd = Detection(arguments)
    ocr = OCR()
    # ner = ner_obj()
    # yolo = Detection()
    # facenet = facenet_obj()
    imagenet = imagenet_obj()
    arc_face = face_model.FaceModel(cfg)

    es = cmd_connect_es()

    # log
    logger = log()

    cfg = Config()
    url_checklist = load_json_file(os.path.join(cfg.url_checklist_path, '{0}_checklist.json'.format(cfg.index)))

    args_dict = {
        "OCD": ocd,
        "OCR": ocr,
        # "ner": ner,
        # "yolo": yolo,
        # "facenet": facenet,
        "arcface": arc_face,
        "imagenet": imagenet,
        "es_obj": es,
        "config": cfg,
        "url_checklist": url_checklist,
        "index": cfg.index,
        "logger": logger,
        "store_path": cfg.store_path
    }
    application = tornado.web.Application([(r"/", MainHandler, dict(args_dict=args_dict))])
    # http_server = tornado.httpserver.HTTPServer(application, max_buffer_size=100000)  # default = 100M
    http_server = tornado.httpserver.HTTPServer(application)
    return http_server


def main(configure):
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # sess = tf.Session(config=tf_config)
    # service
    serv = make_app()

    # 設定 port 為 8080
    serv.listen(9528)

    print("server start")

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main(sys.argv[0])
