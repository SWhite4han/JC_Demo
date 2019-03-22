"""Functions for building the face recognition network.
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import numpy as np
from scipy import misc
import re
from tensorflow.python.platform import gfile
import cv2
from PIL import Image, ImageDraw, ImageFont

from face_module import detect_face
from face_module import config


class facenet_obj(object):
    def __init__(self, allow_growth=True):
        self.facenet_model_path = config.facenet_model_path
        self.mtcnn_margin = config.mtcnn_margin
        self.mtcnn_image_size = config.mtcnn_image_size
        self.mtcnn_minsize = config.mtcnn_minsize  # minimum size of face
        self.mtcnn_threshold = config.mtcnn_threshold  # three steps's threshold
        self.mtcnn_factor = config.mtcnn_factor  # scale factor
        # self.args_seed = config.args_seed

        st = time.time()
        sess = tf.get_default_session()
        if isinstance(sess, type(None)):
            gpu_options = tf.GPUOptions(allow_growth=allow_growth)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        # mtcnn
        sess, self.pnet, self.rnet, self.onet = self._create_mtcnn_for_sess(sess, config.mtcnn_model_path)

        # facenet
        sess, self.facenet_images_placeholder, self.facenet_embeddings, self.facenet_phase_train_placeholder = self._create_facenet_for_sess(sess, config.facenet_model_path)
        self.fnet = lambda img_batch: sess.run(self.facenet_embeddings, feed_dict={self.facenet_images_placeholder: img_batch,
                                                                                   self.facenet_phase_train_placeholder: False})
        print('Load models spend:{0}s'.format(time.time() - st))

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def crop(self, image, random_crop, image_size):
        if image.shape[1]>image_size:
            sz1 = int(image.shape[1]//2)
            sz2 = int(image_size//2)
            if random_crop:
                diff = sz1-sz2
                (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
            else:
                (h, v) = (0,0)
            image = image[(sz1-sz2+v):(sz1+sz2+v), (sz1-sz2+h):(sz1+sz2+h), :]
        return image

    def flip(self, image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image

    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def load_model(self, model):
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = self.get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    def get_model_filenames(self, model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    def get_face_vec(self, face_imgs):
        # Run forward pass to calculate embeddings
        print('Calculating features for images')

        # Batch
        face_vectors = list()
        for img_batch in self.batch(face_imgs, batch_size=50):
            print('processing images')
            img_batch_processed = self.face_process(img_batch, False, False, self.mtcnn_image_size)
            print('calculate face vectors')

            emb_array = self.fnet(img_batch_processed)

            face_vectors += list(emb_array)
        return face_vectors

    def img_read(self, image_path):
        try:
            image = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
            raise
        else:
            # make sure that all images are normal
            # ----------------------------------------------
            if image.ndim < 2:  # an normal image should has at least two dimension(width and high)
                print('Unable to align "%s"' % image_path)
                return
            if image.ndim == 2:  # an image which has only one channel
                img = self.to_rgb(image)
                image = img[:, :, 0:3]
            if image.ndim == 3:  # an image which may have more then three channels
                image = image[:, :, 0:3]
            # ----------------------------------------------
        return image

    def cal_euclidean(self, x, y):
        # x, y must be matrices
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        diff = np.subtract(x, y)
        dist = np.sum(np.square(diff), 1)
        return dist

    def get_face_img(self, pnet, rnet, onet, img_paths):
        # input a list of paths of images
        # return
        #     (1) close-ups of faces
        #     (2) source
        #     (3) locations
        face_closeups = list()
        face_source = list()
        face_locations = list()

        for path in img_paths:
            try:
                img = self.img_read(path)
            except AttributeError as ae:
                print(ae)
                print(path)
            try:
                bounding_boxes, _ = detect_face.detect_face(img, self.mtcnn_minsize, pnet, rnet, onet, self.mtcnn_threshold, self.mtcnn_factor)
            except Exception as e:
                print(path)
                # misc.imshow(img)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]

                for det_no in range(nrof_faces):
                    each_det = np.squeeze(det[det_no])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(each_det[0] - self.mtcnn_margin / 2, 0)  # left Bound
                    bb[1] = np.maximum(each_det[1] - self.mtcnn_margin / 2, 0)  # upper Bound
                    bb[2] = np.minimum(each_det[2] + self.mtcnn_margin / 2, img_size[1])  # right Bound
                    bb[3] = np.minimum(each_det[3] + self.mtcnn_margin / 2, img_size[0])  # lower Bound

                    # Drop face if face size < 60 * 60 pixel.
                    # if max((bb[3] - bb[1]), (bb[2] - bb[0])) < 50:
                    #     continue
                    if (bb[2] - bb[0]) < 50:
                        continue
                    if (bb[3] - bb[1]) < 50:
                        continue

                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = misc.imresize(cropped, (self.mtcnn_image_size, self.mtcnn_image_size), interp='bilinear')

                    face_closeups.append(scaled)
                    face_source.append(path)
                    face_locations.append(bb)

        return face_closeups, face_source, face_locations

    def new_get_face_img(self, pnet, rnet, onet, imgs):
        # input a list of images
        # return
        #     (1) close-ups of faces
        #     (2) source
        #     (3) locations
        face_closeups = list()
        face_source = list()
        face_locations = list()

        for img in imgs:
            # try:
            #     bounding_boxes, _ = detect_face.detect_face(img, self.mtcnn_minsize, pnet, rnet, onet, self.mtcnn_threshold, self.mtcnn_factor)
            # except Exception as e:
            #     misc.imshow(img)
            bounding_boxes, _ = detect_face.detect_face(img, self.mtcnn_minsize, pnet, rnet, onet, self.mtcnn_threshold,
                                                        self.mtcnn_factor)

            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]

                for det_no in range(nrof_faces):
                    each_det = np.squeeze(det[det_no])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(each_det[0] - self.mtcnn_margin / 2, 0)  # left Bound
                    bb[1] = np.maximum(each_det[1] - self.mtcnn_margin / 2, 0)  # upper Bound
                    bb[2] = np.minimum(each_det[2] + self.mtcnn_margin / 2, img_size[1])  # right Bound
                    bb[3] = np.minimum(each_det[3] + self.mtcnn_margin / 2, img_size[0])  # lower Bound

                    # Drop face if face size < 60 * 60 pixel.
                    # if max((bb[3] - bb[1]), (bb[2] - bb[0])) < 50:
                    #     continue
                    if (bb[2] - bb[0]) < 50:
                        continue
                    if (bb[3] - bb[1]) < 50:
                        continue

                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = misc.imresize(cropped, (self.mtcnn_image_size, self.mtcnn_image_size), interp='bilinear')

                    face_closeups.append(scaled)
                    face_source.append(img)
                    face_locations.append(bb)

        return face_closeups, face_source, face_locations

    def face_process(self, face_closeups, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
        # input a list of faces
        # return a nd-array of pre-processed faces
        nrof_samples = len(face_closeups)
        images = np.zeros((nrof_samples, image_size, image_size, 3))
        for i in range(nrof_samples):
            img = face_closeups[i]
            try:
                if img.ndim == 2:
                    img = self.to_rgb(img)
                if do_prewhiten:
                    img = self.prewhiten(img)
                img = self.crop(img, do_random_crop, image_size)
                img = self.flip(img, do_random_flip)
                images[i, :, :, :] = img
            except:
                continue
        return images

    def batch(self, iterable, batch_size=1):
        l = len(iterable)
        for ndx in range(0, l, batch_size):
            yield iterable[ndx:min(ndx + batch_size, l)]

    def _create_mtcnn_for_sess(self, sess, model_path):
        pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)
        return sess, pnet, rnet, onet

    def _create_facenet_for_sess(self, sess, model_path):
        # Load the model
        print('Loading feature extraction model')
        START = time.time()
        self.load_model(model_path)
        print("spent %f sec" % (time.time()-START))

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        return sess, images_placeholder, embeddings, phase_train_placeholder

    def faceDB(self, db_name, img_path=None, update=False):
        code_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(code_path, db_name)):
            os.mkdir(os.path.join(code_path, db_name))
        if update and img_path is None:
            print('if update flag is true, img_path can not be None')
            exit()
        if update or \
                not os.path.exists(os.path.join(code_path, db_name, 'face_vectors.npy')) or \
                not os.path.exists(os.path.join(code_path, db_name, 'face_source.npy')) or \
                not os.path.exists(os.path.join(code_path, db_name, 'face_locations.npy')):
            people_list = os.listdir(img_path)

            print('preparing image paths')
            image_paths = list()
            for person in people_list:
                image_paths += [img_path+'%s/' % (person) + image for image in os.listdir(img_path+'%s/' % person)]

            print('loading images')

            face_closeups, face_source, face_locations = self.get_face_img(image_paths)

            print('processing images')
            input_queue = tf.train.slice_input_producer(face_closeups, shuffle=False)
            image_batch = tf.train.batch(input_queue, batch_size=5)

            # processed_face_closeups = face_process(face_closeups, False, False, mtcnn_image_size)
            processed_face_closeups = self.face_process(image_batch, False, False, self.mtcnn_image_size)

            print('calculate face vectors')
            face_vectors = self.get_face_vec(processed_face_closeups)

            np.save(os.path.join(code_path, db_name, 'face_vectors.npy'), face_vectors)
            np.save(os.path.join(code_path, db_name, 'face_source.npy'), face_source)
            np.save(os.path.join(code_path, db_name, 'face_locations.npy'), face_locations)

            # show
            # for count in range(len(face_vectors)):
            #     misc.imshow(face_closeups[count])
            #     misc.imshow(misc.imread(face_source[count]))
            #     bb = face_locations[count]
            #     misc.imshow(misc.imread(face_source[count])[bb[1]:bb[3], bb[0]:bb[2], :])
        else:
            face_vectors = np.load(os.path.join(code_path, db_name, 'face_vectors.npy'))
            face_source = np.load(os.path.join(code_path, db_name, 'face_source.npy'))
            face_locations = np.load(os.path.join(code_path, db_name, 'face_locations.npy'))

        return face_vectors, face_source, face_locations

    def puttext_in_chinese(self, img, text, location):
        # cv2 to pil
        cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img)

        # text
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("simhei.ttf", 10, encoding="utf-8")
        draw.text(location, text, (255, 0, 0), font=font)  # third parameter is color

        # pil to cv2
        cv2_text_im = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv2_text_im

    def drawBoundaryBox(self, face_sources, face_locations, person_names, distances):
        # font style
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale = 0.5
        # fontColor = (255, 255, 255)
        # lineType = 2

        imgList = list()
        face_counter_for_each_image = 0
        for face_no in range(len(person_names)):
            source = face_sources[face_no]
            location = face_locations[face_no]
            name = person_names[face_no]
            distance = distances[face_no]

            if type(source) == np.ndarray:
                img = source
            else:
                img = cv2.imread(source)

            # check whether those faces are in the same image or not
            try:
                if not np.all(pre_img == img):
                    if face_counter_for_each_image > 0:
                        imgList.append(marked_img)
                        pre_img = img
                        marked_img = img.copy()
                    face_counter_for_each_image = 0
            except:
                pre_img = img
                marked_img = img.copy()

            # draw boundary box
            cv2.rectangle(marked_img, (location[0], location[1]), (location[2], location[3]), (0, 255, 0), 2)
            # cv2.putText(img, '%s: %.3f' % (name, distance), (location[0], location[3]), font, fontScale, fontColor, lineType)
            marked_img = self.puttext_in_chinese(marked_img, '%s: %.3f' % (name, distance), (location[0], location[3]))

            if face_no == len(person_names)-1:
                imgList.append(marked_img)

            face_counter_for_each_image += 1

        return imgList

    def face2vec(self, image_paths):
        sess = tf.get_default_session()

        print('loading images')

        # mtcnn
        face_closeups, face_source, face_locations = self.get_face_img(self.pnet, self.rnet, self.onet, img_paths=image_paths)
        print('len face_closeups = {0}'.format(len(face_closeups)))

        # facenet
        face_vectors = self.get_face_vec(face_imgs=face_closeups)

        return face_vectors, face_source, face_locations

    def new_face2vec(self, images):
        # mtcnn
        face_closeups, face_source, face_locations = self.new_get_face_img(self.pnet, self.rnet, self.onet, imgs=images)
        print('len face_closeups = {0}'.format(len(face_closeups)))

        # facenet
        face_vectors = self.get_face_vec(face_imgs=face_closeups)

        return face_vectors, face_source, face_locations


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
