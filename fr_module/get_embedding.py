import os
from fr_module import face_model
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-dir', default="./test_img", help='')
parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--model', default='model_50/model-0000, 0', help='path to load model.')
parser.add_argument('--model', default='model_alignt_person/model, 792', help='path to load model.') 
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parsers = parser.parse_args()

def get_images(input_images):
    """
    input: (1) path of directory of many people's face image directory named by face owners.
           (2) path of directory of face images belong to one person named by the face owner.
           (3) path of one image file
    return: dictionary with person name as "key" and list of paths of images as "value"
    """
    # print "image source: %s" % input_images
    face_dict = dict()
    # input_images is a directory contain many face images or many directorys with face image in it.
    if os.path.isdir(input_images):
        element_list = os.listdir(input_images)
        # print "contain:" 
        # print element_list
        # element may be image file or a directory of images named by the face owner of images
        for element in element_list:
            element_path = os.path.join(input_images, element)
            # directory: element is person name
            if os.path.isdir(element_path):
                face_dict[element] = dict()
                file_list = os.listdir(element_path)
                for filename in file_list:
                    file_path = os.path.join(element_path, filename)
                    face_dict[element][file_path] = -1
            # image file: element is image file name
            else:
                person_name = input_images.split('/')[-1]
                if person_name not in face_dict:
                    face_dict[person_name] = dict()
                face_dict[person_name][os.path.join(input_images, element)] = -1
    # input_images is one image file
    else:
        face_dict['unknown'] = dict()
        face_dict['unknown'][input_images] = -1

    return face_dict


if __name__ == "__main__":
    # args = parser.parse_args()
    from configuration.config import Config
    args = Config()
    model = face_model.FaceModel(args)
    # face_dict = get_images(os.path.abspath(os.path.join(os.path.curdir, "..", args.image_dir)))
    face_dict = get_images(os.path.abspath('/home/ainb/data'))
    # print face_dict
    for person in face_dict.keys():
        for raw_img_path in face_dict[person].keys():
            raw_img = cv2.imread(raw_img_path)
            aligned_img = model.get_input(raw_img)
            face_dict[person][raw_img_path] = model.get_feature(aligned_img)
    # print face_dict

    # save dictionary
    # np.save("db.npy", face_dict)
    # print "db saved"

    # load db
    # db_from_npy = np.load("db.npy")
    # print(db_from_npy.item().keys())

    # get feature all vectors of the specified person
    # face_dict[<person_name>].values()

    # compute similarity of two feature
    # np.dot(f1, f2.T)

    # compute similarity of new feature vector and the specified person
    # np.dot(nf, np.asarray(face_dict[<person_name>].values()))

