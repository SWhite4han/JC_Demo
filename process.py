import os
import time
import json
import cv2
import numpy as np
import multiprocessing
import tensorflow as tf

from Common.common_lib import batch, get_images
from nlp_module import cc
from nlp_module.ner.eval import ner_obj

from face_module.facenet import facenet_obj
from face_module.face2vec import face2vec_for_query
from obj_dectect_module.Detection import Detection
from image_vec.img2vec import imagenet_obj
from ocr_module.chinese_ocr.eval import OCD, OCR
from PIL import Image


def save_json():
    """

    :return:
    """
    print()


def face_2_vec(facenet, test_img_path, save_path=None):
    img_path = get_images(test_img_path)

    st = time.time()
    detector = Detection()
    print('Load model spent:{0}s'.format(time.time() - st))
    # detection = detector.detect_by_path(img_path=os.path.join(detector.execution_path, "image2.jpg"))
    sources, keywords = detector.detect_by_path_batch(imgs_path=img_path)
    print('Total detecte time:{0}s'.format(time.time() - st))

    # --------------------------------
    # ------ save yolo keywords ------
    # --------------------------------
    save_json()
    # --------------------------------

    have_face = list()
    for idx, keys in enumerate(keywords):
        if 'person' in keys:
            have_face.append(sources[idx])
    print()
    db_face_vectors, db_face_source, _ = facenet.face2vec(image_paths=have_face)

    # ---- Start:Save json file -----
    if save_path:
        # if not os.path.exists(conf.out_vec_path):
        #     os.makedirs(conf.out_vec_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tmp = ''
        count_same_img = 1
        for i in range(len(db_face_vectors)):
            source_path = db_face_source[i]
            if tmp == db_face_source[i]:
                count_same_img += 1
            else:
                tmp = db_face_source[i]
                count_same_img = 1
            sp = source_path.split('/')
            file_name = sp[-2] + '_' + sp[-1].split('.')[0] + '_face_' + str(count_same_img) + '.json'
            with open(os.path.join(save_path, file_name), 'w') as wf:
                json.dump({'imgVec': db_face_vectors[i].tolist(), 'category': 'face', 'imgPath': source_path}, wf,
                          ensure_ascii=False)
    # ---- End:Save json file  -----
    # return {'imgVec': db_face_vectors[i].tolist(), 'category': 'face', 'imgPath': source_path}

    # show
    # for count in range(len(db_face_vectors)):
    #     misc.imshow(misc.imread(db_face_source[count]))
    #     bb = face_locations[count]
    #     misc.imshow(misc.imread(db_face_source[count])[bb[1]:bb[3], bb[0]:bb[2], :])


def img_2_vec(imagenet, img_path, save_path):
    imgs = get_images(img_path)
    img_paths, vecs = imagenet.img_vec(img_paths=imgs)

    # ---- Start:Save json file -----
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(len(vecs)):
            source_path = img_paths[i]
            sp = source_path.split('/')
            file_name = sp[-2] + '_' + sp[-1].split('.')[0] + '_img' + '.json'
            with open(os.path.join(save_path, file_name), 'w') as wf:
                json.dump({'imgVec': vecs[i].tolist(), 'category': 'img', 'imgPath': source_path}, wf, ensure_ascii=False)
    # ---- End:Save json file  -----


def read_net_war():
    import csv
    path_root = '/home/c11tch/workspace/PycharmProjects/JC_Demo/TEST_data_for_infinity/net_war_data'
    info_file = 'persondata1113.csv'
    with open(os.path.join(path_root, info_file), newline='') as csvfile:
        rows = csv.DictReader(csvfile)

        img_paths = list()
        intros = list()
        for row in rows:
            if row['ImageLink'] != 'NULL':
                each_info = ''

                intro_1 = row['Introduction1']
                intro_2 = row['Introduction2']
                if intro_1 != 'NULL':
                    each_info += intro_1 + '\n'

                if intro_2 != 'NULL':
                    each_info += intro_2 + '\n'

                if each_info:
                    p = row['ImageLink'].replace('~/Content/PeoplePhoto//', path_root + '/PeoplePhoto/')
                    img_paths.append(p.replace('\\', '/'))
                    # All type of keywords must be traditional
                    intros.append(cc.trans_s2t(each_info))
    return intros, img_paths


# def net_war_data(img_paths, intros, save_path):
#     if len(intros) > 0:
#         ner_rlt = evaluate_lines_by_call(intros)
#         if save_path:
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#
#             for i in range(len(img_paths)):
#                 sp = img_paths[i].split('/')
#                 file_name = sp[-2] + '_' + sp[-1].split('.')[0] + '_key' + '.json'
#                 with open(os.path.join(save_path, file_name), 'w') as wf:
#                     json.dump({'keywords': ner_rlt[i], 'category': 'keyword', 'imgPath': img_paths[i]}, wf, ensure_ascii=False)
#
#
# def text2ner():
#     # This function can process non empty single sentence or multiple sentences in a list.
#     rlts = evaluate_lines_by_call(['嘉義縣水上鄉公所村幹事侯姓女子承辦選務'])
#     # rlts = evaluate_lines_by_call(['',
#     #                                '嘉義縣水上鄉公所村幹事侯姓女子承辦選務，18日到印刷廠清點選票，\n離開時被警員發現她帶著300多張選票，立刻逮人。',
#     #                                'La new熊今年團隊薪資確定維持平盤，而熊隊今年簽下3位退伍的代訓選手林泓育(見圖)、何信賢及陳雁風，林泓育的簽約金也以略高於400萬元',
#     #                                '藝人孫鵬、狄鶯之子孫安佐被控在美國非法持有彈藥，被關押239天，該案於台北時間昨晚11時在費城的賓州東區聯邦地區法院開庭',
#     #                                '美中贸易战延烧至今，已陆续有在大陆设厂的外国企业出走。尽管目前处于90天暂时停战期，但有媒体报导说，香港玩具制造商考虑将工厂迁出中国大陆，大陆玩具工厂已比鼎盛时期减少了三分之一。',
#     #                                cc.trans_s2t('美中贸易战延烧至今，已陆续有在大陆设厂的外国企业出走。尽管目前处于90天暂时停战期，但有媒体报导说，香港玩具制造商考虑将工厂迁出中国大陆，大陆玩具工厂已比鼎盛时期减少了三分之一。')])
#     for rlt in rlts:
#         print(rlt)


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # graph = tf.Graph().as_default()
    # gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.40)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    # target_img_path = '/data2/JC/TEST_data_for_infinity/img'
    # target_save_path = '/data2/JC/TEST_data_for_infinity/results'

    # p = multiprocessing.Process(target=face_2_vec, args=(target_img_path, target_save_path))
    # p.start()
    # p.join()
    #
    # p = multiprocessing.Process(target=img_2_vec, args=(target_img_path, target_save_path))
    # p.start()
    # p.join()

    # _ = face_2_vec(target_img_path, target_save_path)
    # img_2_vec(target_img_path, target_save_path)

    # ------------------------------- For Net War Data ---------------------------------------------------
    # target_img_path = '/data2/JC/TEST_data_for_infinity/net_war_data/PeoplePhoto'
    # target_save_path = '/data2/JC/TEST_data_for_infinity/results'

    # intro, imgp = read_net_war()
    # p = multiprocessing.Process(target=net_war_data, args=(imgp, intro, target_save_path))
    # p.start()
    # p.join()
    #
    # p = multiprocessing.Process(target=face_2_vec, args=(target_img_path, target_save_path))
    # p.start()
    # p.join()
    #
    # p = multiprocessing.Process(target=img_2_vec, args=(target_img_path, target_save_path))
    # p.start()
    # p.join()
    # _ = face_2_vec(target_img_path, target_save_path)
    # img_2_vec(target_img_path, target_save_path)
    # -----------------------------------------------------------------------------------------------------

    # -------------- IMPORT OBJ(optional) ---------------
    # OCD = OCD("ocr_module/EAST/pretrained_model/east_mixed_149482")
    # OCR = OCR()

    # yolo = Detection()
    # facenet = facenet_obj()
    # imagenet = imagenet_obj()
    # ner = ner_obj()

    # For arc face
    from fr_module import face_model
    from configuration.config import Config

    cfg = Config()
    arc_face = face_model.FaceModel(cfg)

    # -------------- IMPORT OBJ ---------------

    # -------------- START: OCR --------------
    # # image = cv2.imread(r'/mnt/data1/TCH/people_image/金正恩/000189.jpg')
    # image = cv2.imread(r'/mnt/data1/TCH/sol_image_tmp/29.jpg')
    # # image = cv2.imread(r'/mnt/data1/TCH/sol_image_tmp/32.jpg')
    #
    # # EAST
    # # image = cv2.resize(image, (0,0), fx=0.8, fy=0.8)
    # image_list, masked_image, boxes = OCD.detection(image)
    #
    # if image_list:
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
    #         print(cc.trans_s2t("text: %s" % elem['text']))
    #
    #         # cv2 to pil
    #         # cv2_im = cv2.cvtColor(image_list[idx], cv2.COLOR_BGR2RGB)
    #         # pil_im = Image.fromarray(cv2_im)
    #         # pil_im.show()
    #         pass
    # -------------- END: OCR --------------

    # -------------- START: FACE + YOLO -----------
    # ob = face2vec_for_query(yolo, facenet, [cv2.imread(r'/mnt/data1/TCH/people_image/000001.jpg')])
    # print(ob)
    # -------------- END: FACE -------------
    #
    # keys = ner.evaluate_lines_by_call(['中國商務部副部長鍾山前天證實，中美達成初步協議，近期內將不會調高人民幣匯率。'])
    # print(keys)
    # keys = ner.evaluate_lines_by_call(['乒乓球擂台赛首场半决赛战罢刘国梁王晨取得决赛权(附图片1张)本报浙江余姚1月24日电爱立信中国乒乓球擂台赛今天'])
    # print(keys)
    # keys = ner.evaluate_lines_by_call(['中國商務部副部長鍾山前天證實，中美達成初步協議，近期內將不會調高人民幣匯率。'])
    # print(keys)

    # -------------- START: IMAGE -----------
    # _, v = imagenet.img_vec(
    #     [r'/mnt/data1/TCH/people_image/金正恩/000098.jpg'])
    # print(v)
    # -------------- END: IMAGE -------------

    # -------------- TensorBoard ------------
    # graph = tf.get_default_graph()
    # writer = tf.summary.FileWriter("TensorBoard/", graph=graph)
    print()

    # -------------------- mxnet FACE TEST ------------------------
    # face_lst = [
    #     # r'/mnt/data1/TCH/sol_image_tmp/5604.jpg',
    #     r'/mnt/data1/TCH/sol_image_tmp/4906.jpg',
    #     # r'/mnt/data1/TCH/sol_image_tmp/111.jpg',
    #     # r'/mnt/data1/TCH/people_image/金正恩/000098.jpg',
    #     # r'/mnt/data1/TCH/sol_image_tmp/4970.jpg',
    #     # r'/mnt/data1/TCH/sol_image_tmp/4984.jpg',
    # ]
    # for raw_img_path in face_lst:
    #     raw_img = cv2.imread(raw_img_path)
    #     aligned_imgs, bound_box = arc_face.get_multi_input(raw_img)
    #     if aligned_imgs is not None:
    #         for i in range(len(aligned_imgs)):
    #             aligned_img = aligned_imgs[i]
    #             bbox = bound_box[i]
    #             # For showing crouped image
    #             cv2_im = cv2.cvtColor(raw_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], cv2.COLOR_BGR2RGB)
    #             pil_im = Image.fromarray(cv2_im)
    #             pil_im.show()
    #             vec_arc = arc_face.get_feature(aligned_img)
    #             # vec_arc = arc_face.get_feature(np.array(aligned_img.tolist()))
    #             print('arc_face', vec_arc)

    face_lst = [
        r'/home/user/Pictures/sol_image_tmp/5604.jpg',
        # r'/home/user/Pictures/sol_image_tmp/2746.jpg',
        r'/home/user/Pictures/sol_image_tmp/111.jpg',
    ]
    compare = []
    name = []
    for raw_img_path in face_lst:
        raw_img = cv2.imread(raw_img_path)
        aligned_imgs, bound_box = arc_face.get_multi_input(raw_img)
        if aligned_imgs is not None:
            for i in range(len(aligned_imgs)):
                aligned_img = aligned_imgs[i]
                vec_arc = arc_face.get_feature(aligned_img)
                compare.append(vec_arc)
                name.append(raw_img_path)

    from scipy import spatial
    for i in range(len(compare)):
        dist_euclidean = np.linalg.norm(compare[0] - compare[i])
        dist_cos = 1 - spatial.distance.cosine(compare[0], compare[i])
        print(dist_euclidean, dist_cos, name[i])

