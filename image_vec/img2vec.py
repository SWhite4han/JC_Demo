import importlib
import os
import time
import json

import numpy as np
import tensorflow as tf
# from tensorflow.python.tools import inspect_checkpoint as chkp

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3, inception_v3_arg_scope
import skimage
import skimage.io
import skimage.transform

from Common.common_lib import batch_2file, get_images

# -------------------------
try:
    from image_vec.data import *
except Exception as e:
    raise Exception("{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 299, 299
    resized_img = skimage.transform.resize(crop_img, (299, 299))
    return resized_img


def image_process(img):
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 299, 299
    resized_img = skimage.transform.resize(crop_img, (299, 299))
    return resized_img


# def print_prob(prob):
#     synset = class_names
#     # print prob
#     pred = np.argsort(prob)[::-1]
#     # Get top1 label
#     top1 = synset[pred[0]]
#     print("Top1: ", top1, prob[pred[0]])
#     # Get top5 label
#     top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
#     print("Top5: ", top5)
#     return top1


class imagenet_obj(object):
    def __init__(self):
        self.image_batch = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

        # inception_v3
        # --------------------------------------------------------------------
        # network = importlib.import_module('inception_v3')
        with slim.arg_scope(inception_v3_arg_scope()):
            self.prelogits, self.end_points = inception_v3(
                self.image_batch, is_training=False, dropout_keep_prob=1.0, num_classes=1001, reuse=tf.AUTO_REUSE)

        self.PreLogits = self.end_points['PreLogits']
        self.probs = tf.nn.softmax(self.prelogits)
        # --------------------------------------------------------------------
        #
        # # inception_resnet_v2
        # # --------------------------------------------------------------------
        # network = importlib.import_module('inception_resnet_v2')
        # scope = network.inception_resnet_v2_arg_scope(
        #             weight_decay=0.0,
        #             batch_norm_decay=0.995,
        #             batch_norm_epsilon=0.001,
        #             activation_fn=tf.nn.relu
        #         )
        # with slim.arg_scope(scope):
        #     prelogits, _ = network.inception_resnet_v2(
        #         image_batch, is_training=False, dropout_keep_prob=1.0,
        #         num_classes=1001, reuse=None)
        #
        # probs = tf.nn.softmax(prelogits)
        # # --------------------------------------------------------------------

        sess = tf.get_default_session()
        if isinstance(sess, type(None)):
            sess = tf.InteractiveSession()
        self.load_model(sess)

        self.inet = lambda img_batch: sess.run(self.PreLogits,
                                               feed_dict={self.image_batch: img_batch})

    def load_model(self, sess):
        # Initialize variables
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        now_path = os.getcwd()
        print(now_path)
        checkpoint_path = os.path.join(now_path, "image_vec", "pretrained", "inception_v3.ckpt")

        # chkp.print_tensors_in_checkpoint_file(checkpoint_path, tensor_name="", all_tensors=False)

        # model restore
        imagenet_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
        saver = tf.train.Saver(imagenet_scope)

        saver.restore(sess, checkpoint_path)
        print("Model Restored")

    def inference(self, img1):
        # forward pass
        start_time = time.time()
        out_vec = self.inet(img1)

        print("End time : %.5ss" % (time.time() - start_time))
        # print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing

        # -----------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------
        # testing restore model
        #
        # from model_functions import load_model, save_variables_and_metagraph
        # model_name = '20180109-141832'
        # saver = tf.train.Saver()
        # # save model
        # save_path = "./pretrained/tmp/"
        # log_dir = "./pretrained/tmp/logs/"
        # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # step = 0
        # save_variables_and_metagraph(sess, saver, summary_writer, save_path, model_name, step)
        # print("Model saved in file: %s" % save_path)
        #
        # with tf.Session() as sess:
        #     load_model("./pretrained/tmp/")
        #     print("Model Restored")
        #     prob = sess.run(probs, feed_dict={image_batch: img1})
        #     print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing

        # -----------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------
        # saver = tf.train.Saver()
        # saver.save(sess, "./pretrained/tmp/model.ckpt")
        # return out_vec[0][0][0]
        batch_size = out_vec.shape[0]
        out_vec = np.squeeze(out_vec)
        if batch_size == 1:
            out_vec = np.atleast_2d(out_vec)
        return out_vec

    def img_vec(self, img_paths):
        images = list()
        remain_paths = list()
        for path in img_paths:
            try:
                new_array = load_image(path)
                new_array = new_array[:, :, :3]
                new_array = new_array.reshape((1, 299, 299, 3))
            except Exception as e:
                # --- Need to find why error ---
                print(e)
            images.extend(new_array)
            remain_paths.append(path)

        vecs = list()
        imgp = list()

        for img_batch, path_batch in batch_2file(images, img_paths, 20):
            try:
                tmp_vecs = self.inference(np.asarray(img_batch))
                vecs.extend(tmp_vecs)
                imgp.extend(path_batch)
            except Exception as e:
                print(e)
                print(path_batch)
        print(len(vecs))

        return img_paths, vecs

    def new_img_vec(self, imgs):
        images = list()
        for img in imgs:
            # new_array = load_image(path)
            new_array = image_process(img)
            new_array = new_array[:, :, :3]
            new_array = new_array.reshape((1, 299, 299, 3))
            images.extend(new_array)

        vecs = list()
        imgp = list()

        for img_batch, path_batch in batch_2file(images, imgs, 20):
            try:
                tmp_vecs = self.inference(np.asarray(img_batch))
                vecs.extend(tmp_vecs)
                imgp.extend(path_batch)
            except Exception as e:
                print(e)
                # print(path_batch)
        print('len of img vecs', len(vecs))
        print('len of imgs', len(imgs))

        return imgs, vecs


if __name__ == '__main__':
    # img_paths = get_images('/home/c11tch/workspace/PycharmProjects/JC_Demo/TEST_data_for_infinity/img')
    # save_path = '/home/c11tch/workspace/PycharmProjects/JC_Demo/TEST_data_for_infinity/results/i2v_test'

    # images = list()
    # for path in img_paths:
    #     new_array = load_image(path)
    #     new_array = new_array.reshape((1, 299, 299, 3))
    #     images.extend(new_array)
    #
    # start = 0
    # vecs = list()
    # imgp = list()
    # for img_batch, path_batch in batch_2file(images, img_paths, 25):
    #     if start == 0:
    #         tmp_vecs = inference(np.array(img_batch))
    #         start += 1
    #     else:
    #         tmp_vecs = inference(np.array(img_batch), reuse=True)
    #     print(tmp_vecs)
    #     print(tmp_vecs.shape)
    #     vecs.extend(tmp_vecs)
    #     imgp.extend(path_batch)
    # print(len(vecs))

    # --- Save results ---
    # np.save(os.path.join(save_path, 'img_source.npy'), imgp)
    # np.save(os.path.join(save_path, 'img_vectors.npy'), vecs)

    # --- Test distance between images ---
    # target_vec = vecs[0]
    # rand_idx = np.random.randint(0, len(vecs))
    # target_vec = vecs[rand_idx]
    # for vec in vecs:
    #     print(np.linalg.norm(target_vec - vec))
    # print()

    imagenet = imagenet_obj()
    img_paths, vecs = imagenet.img_vec(img_paths=['/home/c11tch/workspace/PycharmProjects/JC_Demo/TEST_data_for_infinity/img'])
    save_path = '/home/c11tch/workspace/PycharmProjects/JC_Demo/TEST_data_for_infinity/results/'
    # ---- Start:Save json file -----
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(len(vecs)):
            source_path = img_paths[i]
            sp = source_path.split('/')
            file_name = sp[-2] + '_' + sp[-1].split('.')[0] + '_img_vec' + '.json'
            with open(os.path.join(save_path, file_name), 'w') as wf:
                json.dump({'imgVec': vecs[i].tolist()}, wf, ensure_ascii=False)
    # ---- End:Save json file  -----