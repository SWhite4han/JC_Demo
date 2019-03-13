import os
import json
import time
import multiprocessing

from Common.common_lib import cal_days
from nlp_module import cc
# from nlp_module.ner.eval import evaluate_lines_by_call

from face_module.facenet import facenet_obj
from face_module.face2vec import face2vec_for_query
from obj_dectect_module.Detection import Detection
from image_vec.img2vec import imagenet_obj

from TestElasticSearch import InfinitySearchApi


path_root = '/data2/Dslab_News/'
path_roots = [os.path.join(path_root, 'AppleDaily')]
# path_roots = [os.path.join(path_root, 'AppleDaily'),
#               os.path.join(path_root, 'BusinessTimes'),
#               os.path.join(path_root, 'LTN'),
#               os.path.join(path_root, 'ChinaElectronicsNews'),
#               os.path.join(path_root, 'Chinatimes'),
#               os.path.join(path_root, 'DogNews')]
output_path = '/data2/JC/toELK'

if output_path:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

yolo = Detection()
facenet = facenet_obj()
imagenet = imagenet_obj()

def cmd_connect_es():
    try:
        es_obj = InfinitySearchApi.InfinitySearch('hosts=10.10.53.201,10.10.53.204,10.10.53.207;port=9200;id=esadmin;passwd=esadmin@2018')
        status = es_obj.status()
        print(status)
        return es_obj
    except Exception as ex:
        print(ex)


def collect_file_by_date(date, infos):
    selected = ['政治', '娛樂', '國際', '體育', '財經', '地方', '影視', '社會']
    list_2_save = list()
    list_contents = list()
    list_img_paths = list()
    for meta_path in infos:
        print(meta_path)
        if os.path.isfile(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                daily_news = json.load(f)

                for _, news_info in daily_news.items():
                    # Get full path
                    real_path = os.path.join(os.path.split(meta_path)[0], news_info['BigCategory'], news_info['Category'],
                                             '{0}_{1}.json'.format(date, news_info['Title']))

                    try:
                        with open(real_path, 'r', encoding='utf-8') as jf:
                            news = json.load(jf)
                    except Exception as e:
                        print(real_path)
                        print(e)
                        continue

                    if len(news['ImgPath']) > 0:
                        for img_path in news['ImgPath']:
                            index_dict = dict()
                            index_dict['Source'] = news['Source']
                            index_dict['Category'] = news['Category']
                            index_dict['BigCategory'] = news['BigCategory']
                            index_dict['imgPath'] = img_path
                            # one image one dict to save
                            list_2_save.append(index_dict)
                            list_contents.append(news['Title'] + '\n' + news['Content'])
                            list_img_paths.append(img_path)
    return list_2_save, list_contents, list_img_paths


# def save_key_files(list_2_save, contents):
#     keywords = evaluate_lines_by_call(contents)
#     for index, keys in enumerate(keywords):
#         tmp_dict = list_2_save[index]
#         tmp_dict['keywords'] = list(set(keys))
#         tmp_dict['category'] = 'keyword'
#         orl_name = os.path.basename(tmp_dict['imgPath']).split('.')[0]
#         file_name = '%s_%s_%s_%s_key.json' % (date, tmp_dict['Source'], tmp_dict['Category'], orl_name)
#         with open(os.path.join(output_path, file_name), 'w') as wf:
#             json.dump(tmp_dict, wf, ensure_ascii=False, indent=4)
#
#
# def upload_key_files(list_2_save, list_contents, es_obj):
#     keywords = evaluate_lines_by_call(list_contents)
#     for index, keys in enumerate(keywords):
#         tmp_dict = list_2_save[index].copy()
#         # Remove duplicate keywords.
#         tmp_dict['keywords'] = list(set(keys))
#         tmp_dict['category'] = 'keyword'
#         # es_obj.push_data(json.dumps(list_2_save[index], ensure_ascii=False, indent=4))
#         print(tmp_dict)
#         es_obj.push_data(tmp_dict)


def upload_face_files(img_path, es_obj):

    st = time.time()
    print('Load model spent:{0}s'.format(time.time() - st))
    # detection = detector.detect_by_path(img_path=os.path.join(detector.execution_path, "image2.jpg"))
    sources, keywords = yolo.detect_by_path_batch(imgs_path=img_path)
    print('Total detecte time:{0}s'.format(time.time() - st))

    have_face = list()
    for idx, keys in enumerate(keywords):
        if 'person' in keys:
            have_face.append(sources[idx])
    print()
    db_face_vectors, db_face_source, _ = facenet.face2vec(image_paths=have_face)

    if len(db_face_vectors) > 0:
        tmp = ''
        count_same_img = 1
        for i in range(len(db_face_vectors)):
            source_path = db_face_source[i]
            if tmp == db_face_source[i]:
                count_same_img += 1
            else:
                tmp = db_face_source[i]
                count_same_img = 1

            es_obj.push_data({'imgVec': db_face_vectors[i].tolist(),
                              'category': 'face',
                              'imgPath': source_path})


def upload_img_files(img_path, es_obj):
    img_paths, vecs = imagenet.img_vec(img_paths=img_path)

    # ---- Start:Save json file -----
    if len(vecs) > 0:
        for i in range(len(vecs)):
            source_path = img_paths[i]
            vec = vecs[i].tolist()
            if len(vec) == 2048:
                es_obj.push_data({'imgVec': vec,
                                  'category': 'img',
                                  'imgPath': source_path})


def save_img_files(img_path):
    img_paths, vecs = imagenet.img_vec(img_paths=img_path)

    # ---- Start:Save json file -----
    if len(vecs) > 0:
        count = 0
        for i in range(len(vecs)):
            count += 1
            source_path = img_paths[i]
            d = {
                'imgVec': vecs[i].tolist(),
                'category': 'img',
                'imgPath': source_path
            }
            # with open('/tmp/jc_test/%s.json' % count, 'w') as wf:
            #     json.dump(d, wf, ensure_ascii=False)


if __name__ == '__main__':
    # 20180101-20180831 Apple only
    # dates = cal_days('20180207', '20180207') this day face image error
    dates = cal_days('20180107', '20180831')
    es = cmd_connect_es()
    for date in dates:
        meta_info = [os.path.join(p, date, date + '.json') for p in path_roots]
        headers, contents, image_path = collect_file_by_date(date, meta_info)

        # For test
        # upload_key_files(headers, contents, es)
        # upload_face_files(image_path, es)
        upload_img_files(image_path, es)

        # --- Process news and push to ELK. ---
        # p = multiprocessing.Process(target=upload_key_files, args=(headers, contents, es))
        # p.start()
        # p.join()
        #
        # p = multiprocessing.Process(target=upload_face_files, args=(image_path, es))
        # p.start()
        # p.join()

        # p = multiprocessing.Process(target=upload_img_files, args=(image_path, es))
        # p.start()
        # p.join()
        # -------------------------------------

        print()
