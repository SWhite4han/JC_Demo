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
    # img_path = [r'/mnt/data1/TCH/people_image/金正恩/000098.jpg']
    # task = '2'
    # -----------

    # --- FACE ---
    # img_path = [r'/mnt/data1/TCH/people_image/金正恩/000098.jpg']
    # task = '0'
    # ------------

    # --- IMAGE ---
    img_path = [r'/mnt/data1/TCH/people_image/金正恩/000098.jpg']
    task = '1'
    # -------------

    # --- UPLOAD IMAGES ---
    # img_path = r'/mnt/data1/TCH/people_image/small_up_test'
    # task = 'upload_img'
    # -------------

    port = '9528'
    # port = '9527'

    if port == '9528':
        if task == 'upload_img':
            from Common.common_lib import get_images

            img_paths = get_images(img_path)

            post_data = {
                'task': task,
                'img_paths': img_paths,
            }
        else:
            post_data = {
                'task': task,
                'img_paths': img_path,
            }
    else:
        image = cv2.imread(img_path)
        b64 = img2string(image)

        post_data = {
            'task': task,
            'image': b64,
        }

    # --- Test Response ---
    # for i in range(10):
    #     print('batch', i+1)
    #     # rlt = super_post(data=post_data, url='http://10.10.53.209:9527')
    #     rlt = super_post(data=post_data, url='http://192.168.1.88:9528')
    #     rlt_data = json.loads(rlt.text)
    #     print(rlt_data)
    # ---------------------
    rlt = super_post(data=post_data, url='http://192.168.1.100:%s' % port)
    rlt_data = json.loads(rlt.content)
    print(rlt_data)
