import os
import time
import json
from face_module import config
import numpy as np
import tensorflow as tf
import configuration.f2v_config as conf
from scipy import misc


def get_images(data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    # for parent, dirnames, filenames in os.walk(conf.test_data_path):
    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def face2vec_for_query(yolo, facenet, test_img):
    st = time.time()
    print('Load model spent:{0}s'.format(time.time() - st))

    sources, keywords, _ = yolo.detect_by_image_batch(imgs=test_img)

    print('Total detecte time:{0}s'.format(time.time() - st))

    have_face = list()
    for idx, keys in enumerate(keywords):
        if 'person' in keys:
            have_face.append(sources[idx])
    # have_face.extend(test_img_path)

    if have_face:
        # db_face_vectors, db_face_source, _ = facenet.face2vec(images=have_face)
        db_face_vectors, db_face_source, _ = facenet.new_face2vec(images=have_face)

        return {'imgVec': db_face_vectors[0].tolist(), 'response': 'good'}
    else:
        return {'response': 'no face in image'}


def face2vec_for_sol_data(yolo, facenet, test_img, path):
    st = time.time()
    print('Load model spent:{0}s'.format(time.time() - st))
    # detection = yolo.detect_by_path(img_path=os.path.join(yolo.execution_path, "image2.jpg"))

    # --- Call by image path ---
    # sources, keywords, obj_locations = yolo.detect_by_path_batch(imgs_path=test_img)
    # --- Call by image ndarray ---
    sources, keywords, obj_locations = yolo.detect_by_image_batch(imgs=test_img)

    print('Total detecte time:{0}s'.format(time.time() - st))

    have_face = list()
    locations = list()
    indices = list()
    for idx, keys in enumerate(keywords):
        if 'person' in keys:
            have_face.append(path[idx])
            indices.append(idx)

    if have_face:
        # --- Call by image path ---
        db_face_vectors, db_face_source, face_locations = facenet.face2vec(image_paths=have_face)
        # --- Call by image ndarray ---
        # db_face_vectors, db_face_source, _ = facenet.new_face2vec(images=have_face)

        return db_face_vectors, db_face_source, face_locations
    else:
        return None, None, None


def face2vec_for_query_path(yolo, facenet, test_img):
    st = time.time()
    print('Load model spent:{0}s'.format(time.time() - st))
    # detection = yolo.detect_by_path(img_path=os.path.join(yolo.execution_path, "image2.jpg"))

    # --- Call by image path ---
    sources, keywords, obj_locations = yolo.detect_by_path_batch(imgs_path=test_img)
    # --- Call by image ndarray ---
    # sources, keywords, _ = yolo.detect_by_image_batch(imgs=test_img)

    print('Total detecte time:{0}s'.format(time.time() - st))

    have_face = list()
    locations = list()
    for idx, keys in enumerate(keywords):
        if 'person' in keys:
            have_face.append(sources[idx])
            # This code do not be testing.
            # for i, value in enumerate(keys):
            #     if value == 'person':
            #         locations.append([obj_locations[idx][i] for i, value in enumerate(keys) if value == 'person'])

    if have_face:
        # --- Call by image path ---
        db_face_vectors, db_face_source, face_locations = facenet.face2vec(image_paths=have_face)
        # --- Call by image ndarray ---
        # db_face_vectors, db_face_source, _ = facenet.new_face2vec(images=have_face)

        return db_face_vectors, db_face_source, face_locations
    else:
        return None, None, None


def main():
    from face_module.facenet import facenet_obj
    from obj_dectect_module.Detection import Detection

    yolo = Detection()
    facenet = facenet_obj()

    print('preparing image paths')
    images_list = get_images(conf.test_data_path)

    face2vec_for_query_path(yolo, facenet, test_img=images_list)

    # Get face vector only
    # db_face_vectors, db_face_source, _ = facenet.face2vec(image_paths=images_list, save_path=conf.out_vec_path)

    # For test euclidean distance between all images
    # target = db_face_vectors[4]
    # for v in db_face_vectors:
    #     print(cal_euclidean(target, v))


if __name__ == '__main__':
    main()
