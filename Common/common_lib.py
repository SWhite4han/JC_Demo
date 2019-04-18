import io
import os
import cv2
import time
import base64
import datetime
import requests
import numpy as np
from PIL import Image


def get_images(data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    # for parent, dirnames, filenames in os.walk(conf.test_data_path):
    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def download_image(url, image_file_path='/mnt/data1/TCH/sol_image_tmp'):
    filename = url[url.rfind("/") + 1:]
    suffix_list = ['jpg', 'gif', 'png', 'tif', 'svg', 'pdf', ]

    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    file_type = filename.split('.')[-1]

    if file_type.lower() in suffix_list:
        out_file_path = os.path.join(image_file_path, filename)
        with Image.open(io.BytesIO(r.content)) as im:
            im.save(out_file_path)
        print('Image downloaded from url: {} and saved to: {}.'.format(url, out_file_path))
        return out_file_path


def batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


def batch_2file(iterable, iterable2, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)], iterable2[ndx:min(ndx + batch_size, l)]


def cal_days(begin_date=None, end_date=None, format_in="%Y%m%d", format_out="%Y%m%d"):
    if end_date:
        # 找出start -> end 之間的每一天
        date_list = []
        begin_date = datetime.datetime.strptime(begin_date, format_in)
        end_date = datetime.datetime.strptime(end_date, format_in)
        # 判斷日期先後
        if begin_date > end_date:
            begin_date, end_date = end_date, begin_date

        while begin_date <= end_date:
            date_str = begin_date.strftime(format_out)
            date_list.append(date_str)
            begin_date += datetime.timedelta(days=1)
        return date_list
    else:
        if begin_date:
            return [datetime.datetime.strptime(begin_date, format_in).strftime(format_out)]
        else:
            return [time.strftime(format_out)]


def string2img(encode_string):
    img_string = base64.b64decode(encode_string)
    nparr = np.fromstring(img_string, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def img2string(img):
    _, buffer = cv2.imencode('.jpg', img)
    encode_string = str(base64.b64encode(buffer).decode('utf-8'))
    return encode_string
