from imageai.Detection import ObjectDetection
from Common.common_lib import batch, get_images
import os
import time
import numpy as np
import keras.backend as K

class Detection():

    # key_transform = ["person", "bicycle", "car", "motorcycle", "airplane",
    #                  "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    #                  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    #                  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    #                  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    #                  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    #                  "broccoli", "carrot", "hot dog", "pizza", "donot", "cake", "chair", "couch", "potted plant", "bed",
    #                  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    #                  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer",
    #                  "toothbrush"]

    def __init__(self):
        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(os.path.join(self.execution_path, "obj_dectect_module", "yolo.h5"))
        # self.detector.setModelPath(os.path.join(self.execution_path, "yolo.h5"))
        # self.detector.setModelPath(os.path.join("/home/c11tch/workspace/PycharmProjects/JC_Demo",
        #                                         "obj_dectect_module",
        #                                         "yolo.h5"))
        self.detector.loadModel()

    def detect_by_path(self, img_path):
        # detections = self.detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image2.jpg"),
        #                                                   output_image_path=os.path.join(execution_path, "image2new.jpg"),
        #                                                   minimum_percentage_probability=30)
        detections = self.detector.detectObjectsFromImage(input_image=os.path.join(img_path),
                                                          output_type="array",
                                                          minimum_percentage_probability=30)
        # Only return show_detections words
        return detections[1]

    def detect_by_path_batch(self, imgs_path):
        detection_sources = list()
        detection_keys = list()
        detection_position = list()
        for img in imgs_path:
            detection_sources.append(img)
            tmp_list = list()
            position_list = list()
            for it in self.detector.detectObjectsFromImage(input_image=os.path.join(img),
                                                           output_type="array",
                                                           minimum_percentage_probability=30)[1]:
                tmp_list.append(it['name'])
                position_list.append(it['box_points'])
            detection_keys.append(tmp_list)
            detection_position.append(position_list)
        # Only return show_detections words
        return detection_sources, detection_keys, detection_position

    def detect_by_image_batch(self, imgs):
        detection_sources = list()
        detection_keys = list()
        detection_position = list()
        for img in imgs:
            detection_sources.append(img)
            tmp_list = list()
            position_list = list()
            for it in self.detector.detectObjectsFromImage(input_image=img,
                                                           input_type="array",
                                                           output_type="array",
                                                           minimum_percentage_probability=30)[1]:
                tmp_list.append(it['name'])
                position_list.append(it['box_points'])
            detection_keys.append(tmp_list)
            detection_position.append(position_list)
        # Only return show_detections words
        return detection_sources, detection_keys, detection_position

    @staticmethod
    def show_detections(detections):
        for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
            print("--------------------------------")


if __name__ == '__main__':
    # test_img_path = '/data1/JC_Sample/sample_data_news/img/20180610/國際大事'
    test_img_path = '/media/clliao/006a3168-df49-4b0a-a874-891877a888701/TCH/face_images_verysmall'
    img_path = get_images(test_img_path)

    st = time.time()
    detector = Detection()
    print('Load model spent:{0}s'.format(time.time() - st))
    # detection = detector.detect_by_path(img_path=os.path.join(detector.execution_path, "image2.jpg"))
    sources, keywords = detector.detect_by_path_batch(imgs_path=img_path)
    print('Total detecte time:{0}s'.format(time.time() - st))

    # Save to local
    # np.save(os.path.join('/data1/JC_Sample/sample_data_news/results', 'detect_source.npy'), sources)
    # np.save(os.path.join('/data1/JC_Sample/sample_data_news/results', 'detect_keywords.npy'), keywords)



