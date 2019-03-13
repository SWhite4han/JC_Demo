import importlib
import time

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3, inception_v3_arg_scope
import skimage
import skimage.io
import skimage.transform

import json

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


def print_prob(prob):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def inference(img1, reuse=None):
    image_batch = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    placeholder_reuse = tf.placeholder(tf.bool)

    # inception_v3
    # --------------------------------------------------------------------
    network = importlib.import_module('inception_v3')
    with slim.arg_scope(inception_v3_arg_scope()):
        prelogits, end_points = inception_v3(
                    image_batch, is_training=False, dropout_keep_prob=1.0,
                    num_classes=1001, reuse=reuse)
                    # num_classes=1001, reuse=placeholder_reuse)

    PreLogits = end_points['PreLogits']
    probs = tf.nn.softmax(prelogits)
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

    sess = tf.InteractiveSession()

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # model restore
    saver = tf.train.Saver()
    saver.restore(sess, "./pretrained/inception_v3.ckpt")
    # saver.restore(sess, "./pretrained/inception_resnet_v2_2016_08_30.ckpt")
    print("Model Restored")

    # img1 = load_image("data/puzzle.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    # img1 = load_image("data/tiger1.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    # img1 = load_image("data/laska.png")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    # img1 = load_image("data/ASUS-Zenbo-Front_result.jpg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    # img1 = img1.reshape((1, 299, 299, 3))


    # img1 = misc.imread("data/laska.png")
    # img1 = img1[:, :, 0:3]
    # img1 = misc.imresize(img1, (299, 299, 3))
    # img1 = [img1]
    # img1 = np.array(img1)
    # misc.imshow(img1)


    # forward pass
    start_time = time.time()
    # prob = sess.run(probs, feed_dict={image_batch: img1, placeholder_reuse: None})
    out_vec = sess.run(PreLogits, feed_dict={image_batch: img1, placeholder_reuse: reuse})
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
    return out_vec[0][0][0]

if __name__ == '__main__':
    # img = load_image('data/tiger4.jpeg')
    # img = img.reshape((1, 299, 299, 3))
    # tmp_vec = inference(img)
    # print(tmp_vec)

    vecs = []
    for i in range(3):
        path = 'data/tiger{0}'.format(i+1)
        img = load_image(path + '.jpeg')
        img = img.reshape((1, 299, 299, 3))
        if i == 0:
            tmp_vec = inference(img)
        tmp_vec = inference(img, reuse=True)
        vecs.append(tmp_vec)
        with open(path + '.json', 'w') as f:
            json.dump({'imgName': path.replace('data/', ''), 'imgPath': path, 'imgVec': tmp_vec.tolist()}, f)
        print(tmp_vec)
    for i in range(2):
        path = 'data/cat{0}'.format(i+1)
        img = load_image(path + '.jpg')
        img = img.reshape((1, 299, 299, 3))
        tmp_vec = inference(img, reuse=True)
        vecs.append(tmp_vec)
        with open(path + '.json', 'w') as f:
            json.dump({'imgName': path.replace('data/', ''), 'imgPath': path, 'imgVec': tmp_vec.tolist()}, f)
        print(tmp_vec)
    for i in range(2):
        path = 'data/dog{0}'.format(i+1)
        img = load_image(path + '.jpg')
        img = img.reshape((1, 299, 299, 3))
        tmp_vec = inference(img, reuse=True)
        vecs.append(tmp_vec)
        with open(path + '.json', 'w') as f:
            json.dump({'imgName': path.replace('data/', ''), 'imgPath': path, 'imgVec': tmp_vec.tolist()}, f)
        print(tmp_vec)
    for i in range(2):
        path = 'data/juice{0}'.format(i+1)
        img = load_image(path + '.jpg')
        img = img.reshape((1, 299, 299, 3))
        tmp_vec = inference(img, reuse=True)
        vecs.append(tmp_vec)
        with open(path + '.json', 'w') as f:
            json.dump({'imgName': path.replace('data/', ''), 'imgPath': path, 'imgVec': tmp_vec.tolist()}, f)
        print(tmp_vec)

    print(vecs)
    # target = load_image('data/tiger4.jpeg')
    # target = target.reshape((1, 299, 299, 3))
    # target_vec = inference(target, reuse=True)
    target_vec = vecs[0]
    for vec in vecs:
        print(np.linalg.norm(target_vec - vec))
