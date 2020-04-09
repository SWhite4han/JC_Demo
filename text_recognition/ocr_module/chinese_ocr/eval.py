"""
optical character detection based on EAST
"""
import os
import numpy as np

from PIL import Image
import Levenshtein

from text_recognition.ocr_module.chinese_ocr.ocr_model import crnnRec
# from text_recognition.ocr_module.chinese_ocr.dataset import icdar2003

# class OCR_tess(object):
#     """
#     Optical Character Recognition(tesseract)
#     --list-langs:
#         chi_sim
#         chi_tra
#         eng
#         osd
#     """
#
#     def __init__(self):
#         pass
#
#     def recognize(self, image, lang=None):
#         text = pytesseract.image_to_string(image, lang=lang)
#         return text


class OCR(object):
    """
    image format: PIL
    """
    def __init__(self, DETECTANGLE=True, leftAdjust=True, rightAdjust=True, alph=0.2, ifadjustDegree=False):
        # ocr parameter
        self.detectAngle = DETECTANGLE  # 是否进行文字方向检测
        self.leftAdjust = leftAdjust  # 对检测的文本行进行向左延伸
        self.rightAdjust = rightAdjust  # 对检测的文本行进行向右延伸
        self.alph = alph  # 对检测的文本行进行向右、左延伸的倍数
        self.ifadjustDegree = ifadjustDegree  # 是否先小角度调整文字倾斜角度

    def recognize(self, img, boxes):
        newBox = list()
        for box in boxes:
            new_box = list()
            for coordinate in box:
                new_box.extend(coordinate)
            newBox.append(new_box)

        W, H = img.size

        f = 1.0
        result_list = crnnRec(np.array(img), newBox, self.leftAdjust, self.rightAdjust, self.alph, 1.0 / f)
        return result_list

    def minimum_external_rectangular(self, boxes_list):
        """
        using lines horizontal to axis to represent the bounding box.
        lines are represented by x1, x2, y1, y2

          0_____x1_______x2_______
          |     |        |
        y1|-----O--------O--------
          |     |////////|
          |     |////////|
        y2|-----O--------O--------
          |     |        |

        :param boxes_list:
        :return:
        """
        new_boxes_list = list()
        for box in boxes_list:
            new_boxes_list.append([
                min([box[0][0], box[3][0]]),  # left
                min([box[0][1], box[1][1]]),  # upper
                max([box[1][0], box[2][0]]),  # right
                max([box[2][1], box[3][1]])  # bottom
            ])
        return new_boxes_list

    def eval(self, data_paths, labels, positions_list=None):
        """
        compute accuracy by region of union
        """

        total = 0.
        true = 0.
        I, R, D, E = 0., 0., 0., 0.

        # check dataset
        # -------------
        if len(data_paths) != len(labels):
            print("dataset error")
            return -1
        if isinstance(positions_list, list):
            if len(data_paths) != len(positions_list):
                print("dataset error")
                exit()
        # -------------

        for idx in range(len(data_paths)):
            review_flag = 0
            image_list = list()
            image = Image.open(data_paths[idx])

            # need to crop image by position
            if isinstance(positions_list, list):
                positions = self.minimum_external_rectangular(positions_list[idx])
                for box_idx in range(len(positions)):
                    image_list.append(
                        image.crop((
                            int(positions[box_idx][0]), int(positions[box_idx][1]),
                            int(positions[box_idx][2]), int(positions[box_idx][3])
                        ))
                    )  # left, upper, right, and lower pixel coordinate.
            else:
                image_list.append(image)
                labels[idx] = [labels[idx]]

            # OCR
            result_list = list()
            for img in image_list:
                total += 1
                W, H = img.size
                result = self.recognize(img, np.asarray([[[0, 0], [W, 0], [W, H], [0, H]]]))
                if len(result) == 0:
                    result_list.append('')
                    continue
                result = result[0]['text']
                result_list.append(result)

            for result_idx, result in enumerate(result_list):
                # full match
                if result == labels[idx][result_idx]:
                    true += 1

                # character error rate, CER
                step_list = Levenshtein.opcodes(result.lower(), labels[idx][result_idx].lower())  # pred -> labels[idx]
                # opcodes: return tuple of 5 elements, first means operator, second to fifth means positions of start and end

                for step in step_list:
                    if step[0] == "insert":
                        I += step[4] - step[3]
                        if (step[4] - step[3]) != 0:
                            review_flag = 1
                    elif step[0] == "replace":
                        R += step[2] - step[1]
                        if (step[2] - step[1]) != 0:
                            review_flag = 1
                    elif step[0] == "delete":
                        D += step[2] - step[1]
                        if (step[2] - step[1]) != 0:
                            review_flag = 1
                    elif step[0] == "equal":
                        E += step[2] - step[1]
            if review_flag:
                pass

        print("match accuracy = %f" % (true / total))
        print("CER = %f" % ((R + D + I)/(R + D + E)))
        pass


if __name__ == '__main__':
    pass
# if __name__ == '__main__':
#     OCD = OCD("/home/c11tch/workspace/PycharmProjects/JC_Demo/ocr_module/EAST/pretrained_model/east_mixed_149482/")
#     # OCD = OCD("/data2/relabelled/east_icdar2015_resnet_v1_50_rbox")
#     OCR = OCR()
#
#     # image_path = "/home/c11tch/workspace/PycharmProjects/EAST/data/img_1001.jpg"
#     # image_path = "/home/c11tch/Pictures/123.png"
#     image_path = "/home/c11tch/Pictures/aPICT0034.JPG"
#     # image_path = "/data1/Dataset/OCR/chinese_board/000026.jpg"
#     # image_path = "/data2/EAST_relabelled/1.jpg"
#
#     image = cv2.imread(image_path)
#     # image = Image.open(image_path)
#
#     # EAST
#     image_list, masked_image, boxes = OCD.detection(image)
#
#     # text_list = list()
#
#     # tesseract ocr
#     """
#     OCR_tess = OCR_tess()
#     for idx, img in enumerate(image_list):
#         char_txt = OCR_tess.recognize(img, lang="chi_sim")
#         text_list.append(char_txt)
#
#         # cv2 to pil
#         cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         pil_im = Image.fromarray(cv2_im)
#
#         # show
#         pil_im.show()
#
#         print("text: %s" % char_txt)
#
#     # cv2 to pil, and show
#     cv2_im = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
#     pil_im = Image.fromarray(cv2_im)
#     pil_im.show()
#     # """
#
#     # chineseocr
#     # """
#     # cv2 to pil
#     cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     pil_im = Image.fromarray(cv2_im)
#     img = pil_im.convert("RGB")
#
#     # ocr
#     result_list = OCR.recognize(img, boxes)
#
#     # cv2 to pil, and show
#     cv2_im = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
#     pil_im = Image.fromarray(cv2_im)
#     pil_im.show()
#
#     for idx, elem in enumerate(result_list):
#         print("text: %s" % elem['text'])
#
#         # cv2 to pil
#         cv2_im = cv2.cvtColor(image_list[idx], cv2.COLOR_BGR2RGB)
#         pil_im = Image.fromarray(cv2_im)
#         pil_im.show()
#         pass
#     # """
#
#     # ---------------------------------
#     # dataset = icdar2003(OCD_dataset_path="/data1/Dataset/OCR/icdar2003/Robust Reading and Text Locating/SceneTrialTest",
#     #                     OCR_dataset_path="/data1/Dataset/OCR/icdar2003/Robust Word Recognition/1")
#
#     # OCR eval
#     """
#     data_paths, labels = dataset.OCR_dataset()
#     OCR.eval(data_paths, labels)
#     # """
#
#     # OCD eval
#     """
#     data_paths, labels = dataset.OCD_dataset()
#     precision, recall, f1_score = OCD.eval(data_paths, labels)
#     print("threshold: %.2f" % OCD.IoU_threshold)
#     print("precision: %f" % precision)
#     print("recall: %f" % recall)
#     print("f1 score: %f" % f1_score)
#
#     OCD.sess.close()
#     # """
#
#     # end-to-end eval
#     # """
#     # data_paths, labels, box_positions = dataset.end_to_end_dataset()
#
#     # get crop by ground truth position
#     # OCR.eval(data_paths, labels, box_positions)
#     # """
