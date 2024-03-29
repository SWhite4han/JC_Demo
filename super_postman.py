import cv2
import json
import base64
import requests
from Common.common_lib import string2img


def img2string(img):
    _, buffer = cv2.imencode('.jpg', img)
    encode_string = str(base64.b64encode(buffer).decode('utf-8'))
    return encode_string


def super_post(data, url='http://10.10.53.209:9527'):
    resp = requests.post(url, json=data)
    return resp


if __name__ == '__main__':
    # --- OCR ---
    # # img_path = [r'/mnt/data1/TCH/people_image/金正恩/000098.jpg']
    # img_path = [r'https://s.newtalk.tw/album/news/234/5cb5354330551.JPG']
    # task = '2'
    # -----------

    # --- FACE ---
    # # img_path = [r'/mnt/data1/TCH/people_image/金正恩/000098.jpg']
    # img_path = [
    #     r'https://upload.wikimedia.org/wikipedia/commons/1/1b/%E8%94%A1%E8%8B%B1%E6%96%87%E5%AE%98%E6%96%B9%E5%85%83%E9%A6%96%E8%82%96%E5%83%8F%E7%85%A7.png']
    # task = '0'
    # ------------

    # --- IMAGE ---
    # img_path = [r'/mnt/data1/TCH/people_image/金正恩/000098.jpg']
    # img_path = [r'https://upload.wikimedia.org/wikipedia/commons/1/1b/%E8%94%A1%E8%8B%B1%E6%96%87%E5%AE%98%E6%96%B9%E5%85%83%E9%A6%96%E8%82%96%E5%83%8F%E7%85%A7.png']
    # img_path = [r'https://www.fileformat.info/format/bmp/sample/4cb74cda027a43f3b278c05c3770950f/MARBLES.BMP']
    # img_path = [r'http://210.61.150.56:9000/Content/sci_image/2656/2656.bmp']
    # img_path = [r'http://182.0.0.13/Content/sci_image/6387/6387.jpg']
    # task = '1'
    # -------------

    # --- UPLOAD IMAGES ---
    img_path = r'/mnt/data1/TCH/people_image/small_up_test'
    task = 'upload_img'
    # -------------

    port = '9528'
    # port = '9527'

    if port == '9528':
        if task == 'upload_img':
            from Common.common_lib import get_images

            img_paths = get_images(img_path)

            post_data = {
                'task': task,
                # 'img_paths': img_paths,
                'img_paths': [
                    # r'http://220.133.86.176/TEST/111.jpg',
                    r'http://182.0.0.13/Content/sci_image/3999/3999.jpg',
                    r'12354.321',
                ],
            }
        else:
            post_data = {
                'task': task,
                'img_paths': img_path,
                # 'feature': 4,
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
    # rlt = super_post(data=post_data, url='http://61.216.11.87:%s' % port)  # 院外測試
    rlt = super_post(data=post_data, url='http://182.0.0.200:%s' % port)  # 院內測試
    rlt_data = json.loads(rlt.content)
    print(rlt_data)
    # -------------------------------------------------------------------------

    # ------------------------ For case 4 + 5 face searching ------------------
    # port = '9528'
    # post_data = {
    #     'task': '4',
    #     'img_paths': [
    #         # r'http://210.61.150.56:9000/Content/sci_image/5604/5604.jpg',
    #         r'http://210.61.150.56:9000/Content/sci_image/4906/4906.jpg',
    #     ],
    #     # 'feature': 4,
    # }
    # rlt = super_post(data=post_data, url='http://61.216.11.87:%s' % port)
    # rlt_data = json.loads(rlt.content)
    # print(rlt_data)
    #
    # num_face = rlt_data['total']
    # tmp = []
    # for face_info in rlt_data['face']:
    #     tmp.append(face_info['feature'])
    #
    # # Show image
    # cv2.imshow("123", string2img(rlt_data['face'][0]['pic']))
    # cv2.waitKey(5000)
    #
    # post_data2 = {
    #     'task': '5',
    #     'feature': tmp[0],
    #     "top": 100,
    #     "threshold": 0.01,
    # }
    # rlt = super_post(data=post_data2, url='http://61.216.11.87:%s' % port)
    # rlt_data = json.loads(rlt.content)
    # print(rlt_data)
    # ----------------------------- For case 4 & 5 -----------------------------

