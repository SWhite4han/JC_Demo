import os


class Config(object):
    def __init__(self, exec_path=os.getcwd()):
        self.no_image = os.path.join(exec_path, 'TestElasticSearch', 'no-image.png')
        self.default_path_img = '/data2/Dslab_News/img/'
        self.default_path_ocr = '/data1/Dataset/OCR/'
        self.default_path_face = '/data1/images'

        self.log_path = '/mnt/data1/TCH/tmp_jc/log'
        self.url_checklist_path = '/mnt/data1/TCH/tmp_jc/checklist.json'




