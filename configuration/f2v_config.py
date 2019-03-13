import os
# test_data_path = '/home/c11tch/workspace/PycharmProjects/JC_Demo/ocr_module/EAST/test_data'
test_data_path = '/home/c11tch/workspace/PycharmProjects/JC_Demo/TEST_data_for_infinity/img'
out_vec_path = '/home/c11tch/workspace/PycharmProjects/JC_Demo/TEST_data_for_infinity/results'


# from config.py

src_path, _ = os.path.split(os.path.realpath(__file__))

# args_model = os.path.expanduser(src_path + '/pre_train_models/20170512-110547.pb')
# mtcnn_model = os.path.expanduser(src_path + '/pre_train_models/mtcnn/')
# args_margin = 32
# args_image_size = 160
# minsize = 20  # minimum size of face
# threshold = [0.6, 0.7, 0.7]  # default: [0.6, 0.7, 0.7] # three steps's threshold
# factor = 0.9  # default: 0.709  # scale factor
# args_seed = 666
# nrof_register = 10

# mtcnn
mtcnn_model_path = os.path.expanduser(src_path + '/pre_train_models/mtcnn/')
mtcnn_minsize = 20  # minimum size of face
mtcnn_threshold = [0.6, 0.7, 0.7]  # default: [0.6, 0.7, 0.7] # three steps's threshold
mtcnn_factor = 0.9  # default: 0.709  # scale factor
mtcnn_margin = 32
mtcnn_image_size = 160
mtcnn_batch_size = 5

# facenet
facenet_image_size = 160
facenet_model_path = os.path.expanduser(src_path + '/pre_train_models/20170512-110547.pb')
facenet_batch_size = 5


face_threshold = 0.6

unknown = 'unknown person'

db_name = 'sir_db'
# known_img_path = '/data1/images/0821/manaulLabel/known_people_manually/'
# known_img_path = '/data1/images/0821/manaulLabel/known_people_id/'
known_img_path = '/data1/Dslab_News/JC_Sample/sample_data_include_face/img/AppleDaily/'
# known_img_path = '/data1/images/0821/manaulLabel/big/'
# update = True
update = True
