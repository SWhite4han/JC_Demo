# -*- coding: UTF-8 -*-
import cv2
import threading
import Queue as queue

import dlib

import numpy

from imutils.face_utils import FaceAligner

from imutils.face_utils import rect_to_bb

import imutils
from os import walk
from os.path import join
count = 0
class Worker(threading.Thread):
  def __init__(self, queue, num):
    threading.Thread.__init__(self)
    self.queue = queue
    self.num = num
    self.count = count
  def run(self):
    while self.queue.qsize() > 0:
      # 取得新的資料
      msg = self.queue.get()

      # 處理資料
      print("Worker %d: %s" % (self.num, msg))
      image = cv2.imread(msg)

      image = imutils.resize(image, width=1200)

      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # show the original input image and detect faces in the grayscale

      # image

      rects = detector(gray, 2)

      i = 0
      self.count = self.count + 1
      print(self.count)
      # loop over the face detections

      for rect in rects:
              # extract the ROI of the *original* face, then align the face

              # using facial landmarks

              (x, y, w, h) = rect_to_bb(rect)
              try:
                faceOrig = imutils.resize(image[y:y + h, x:x + w], width=400)
              except:
                continue

              try:
                faceAligned = fa.align(image, gray, rect)
              except:
                continue

              # display the output images

              cv2.imwrite(msg, faceAligned)
              print("save: %s" % (msg))
              i += 1

# 指定要列出所有檔案的目錄
# mypath = "/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/insightface_datasets/tests1"
mypath = "/workspace/datasets/celebrity"


detector = dlib.get_frontal_face_detector()

# predictor = dlib.shape_predictor("/media/clliao/006a3168-df49-4b0a-a874-891877a888701/insightface/src/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("/workspace/src/shape_predictor_68_face_landmarks.dat")


fa = FaceAligner(predictor, desiredFaceWidth=256)
my_queue = queue.Queue()

# load the input image, resize it, and convert it to grayscale
for root, dirs, files in walk(mypath):
  for f in files:
    if f.split('.')[-1] =='jpg':
        try:
                fullpath = join(root, f)
                my_queue.put(fullpath)
                # for i in range(1):
                #         threads[i].join()
        except:
            print("error:"+fullpath)
my_worker = list()
for i in range(1,32):
    my_worker.append(Worker(my_queue, i))

# my_worker2 = Worker(my_queue, 2)
# my_worker3 = Worker(my_queue, 3)
# my_worker4 = Worker(my_queue, 4)



# 讓 Worker 開始處理資料
for i in range(0,31):
    my_worker[i].start()
# my_worker2.start()
# my_worker3.start()
# my_worker4.start()
