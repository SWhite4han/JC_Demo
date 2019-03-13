import numpy as np
from PIL import Image
from ocr_module.chinese_ocr.crnn.crnn import crnnOcr as crnnOcr
from ocr_module.chinese_ocr.apphelper.image import rotate_cut_img, xy_rotate_box, solve


def crnnRec(im, boxes, leftAdjust=False, rightAdjust=False, alph=0.2, f=1.0):
   results = []
   im = Image.fromarray(im) 
   for index,box in enumerate(boxes):

       degree,w,h,cx,cy = solve(box)
       partImg,newW,newH = rotate_cut_img(im,degree,box,w,h,leftAdjust,rightAdjust,alph)
       newBox = xy_rotate_box(cx,cy,newW,newH,degree)
       partImg_ = partImg.convert('L')
       simPred = crnnOcr(partImg_)
       if simPred.strip() != u'':
            results.append({'cx': cx*f, 'cy': cy*f, 'text': simPred, 'w': newW*f, 'h': newH*f, 'degree': degree*180.0/np.pi})
 
   return results

