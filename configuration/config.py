import os


class Config(object):
    def __init__(self, exec_path=os.getcwd()):
        self.no_image = os.path.join(exec_path, 'TestElasticSearch', 'no-image.png')
        self.default_path_img = '/data2/Dslab_News/img/'
        self.default_path_ocr = '/data1/Dataset/OCR/'
        self.default_path_face = '/data1/images'

        # Remove this file if you removed index from elk.
        self.index = 'ui_test_arcface'  # 院內
        # self.index = 'ui_test'  # 院外

        self.log_path = '/mnt/data1/TCH/tmp_jc/log'
        self.url_checklist_path = 'checklist'
        self.store_path = '/home/user/Pictures/sol_image_tmp'

        # For arc face
        self.image_dir = "./test_img"
        self.image_size = "112,112"
        self.model = "fr_module/model_alignt_person/model, 792"  # path to load model.
        self.ga_model = ""  # path to load model.
        self.gpu = 0  # gpu id
        self.det = 0  # mtcnn option, 1 means using R+O, 0means detect from begining
        self.flip = 0  # whether do lr flip aug
        self.threshold = 1.24  # ver dist threshold


