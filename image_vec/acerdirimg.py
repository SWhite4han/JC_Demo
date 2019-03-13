import cv2
import base64
import os
import sys
import requests

def img2string(img):
    ret, buffer = cv2.imencode('.jpg', img)
    encode_string = str(base64.b64encode(buffer).decode('utf-8'))
    return encode_string

def get_images(test_data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def main():
    im_fn_list = get_images(sys.argv[1])
    print(im_fn_list)
    for im_fn in im_fn_list:
        img = cv2.imread(im_fn)
        encode_img = img2string(img)
        postdata = {'image_string': encode_img}
        r = requests.post("http://60.250.226.78:6001/SendImg", data=postdata)
        print(r.text)
if __name__ == '__main__':
    main()