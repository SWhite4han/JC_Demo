import cv2
import json
import base64
import requests


def img2string(img):
    _, buffer = cv2.imencode('.jpg', img)
    encode_string = str(base64.b64encode(buffer).decode('utf-8'))
    return encode_string


def super_post(data, url='http://10.10.53.209:9527'):
    resp = requests.post(url, json=data)
    return resp


if __name__ == '__main__':
    # --- OCR ---
    # img_path = r'/home/c11tch/Pictures/aPICT0034.JPG'
    # task = '2'
    # -----------

    # --- FACE ---
    # img_path = r'/data1/images/川普/google/000004.jpg'
    # task = '0'
    # ------------

    # --- IMAGE ---
    # img_path = r'/data2/Dslab_News/img/AppleDaily/20181228/蘋果國際/國際頭條/20181228_「老美不爽當世界警察」川普伊拉克勞軍發牢騷 酸各國佔便宜/LA21_001.jpg'
    # task = '1'
    # -------------

    # --- UPLOAD IMAGES ---
    img_path = r'/media/clliao/006a3168-df49-4b0a-a874-891877a888701/TCH/face_images_small'
    task = 'upload_img'
    # -------------

    if task == 'upload_img':
        from Common.common_lib import get_images

        img_paths = get_images(img_path)

        post_data = {
            'task': task,
            'img_paths': img_paths,
        }
    else:
        image = cv2.imread(img_path)
        b64 = img2string(image)

        post_data = {
            'task': task,
            'image': b64,
        }

    # --- Test Response ---
    for i in range(10):
        print('batch', i+1)
        # rlt = super_post(data=post_data, url='http://10.10.53.209:9527')
        rlt = super_post(data=post_data, url='http://10.10.53.205:9528')
        rlt_data = json.loads(rlt.text)
        print(rlt_data)
    # ---------------------
