import numpy as np
from numpy import cos, sin


def solve(box):
     """
     绕 cx,cy点 w,h 旋转 angle 的坐标
     x = cx-w/2
     y = cy-h/2
     x1-cx = -w/2*cos(angle) +h/2*sin(angle)
     y1 -cy= -w/2*sin(angle) -h/2*cos(angle)
     
     h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
     w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
     (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

     """
     x1,y1,x2,y2,x3,y3,x4,y4= box[:8]
     cx = (x1+x3+x2+x4)/4.0
     cy = (y1+y3+y4+y2)/4.0  
     w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
     h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2   
     #x = cx-w/2
     #y = cy-h/2
     
     sinA = (h*(x1-cx)-w*(y1 -cy))*1.0/(h*h+w*w)*2
     if abs(sinA)>1:
            angle = None
     else:
        angle = np.arcsin(sinA)
     return angle, w, h, cx, cy


def xy_rotate_box(cx, cy, w, h, angle):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """
    cx    = float(cx)
    cy    = float(cy)
    w     = float(w)
    h     = float(h)
    angle = float(angle)
    x1,y1 = rotate(cx-w/2,cy-h/2,angle,cx,cy)
    x2,y2 = rotate(cx+w/2,cy-h/2,angle,cx,cy)
    x3,y3 = rotate(cx+w/2,cy+h/2,angle,cx,cy)
    x4,y4 = rotate(cx-w/2,cy+h/2,angle,cx,cy)
    return x1, y1, x2, y2, x3, y3, x4, y4


def rotate(x,y,angle,cx,cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    #angle = angle*pi/180
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*cos(angle)+cy
    return x_new,y_new


def rotate_cut_img(im, degree, box, w, h, leftAdjust=False, rightAdjust=False, alph=0.2):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    x_center, y_center = np.mean([x1, x2, x3, x4]), np.mean([y1, y2, y3, y4])
    degree = degree*180.0/np.pi
    right = 0
    left = 0
    if rightAdjust:
        right = 1
    if leftAdjust:
        left = 1
    
    box = (max(1, x_center-w/2-left*alph*(w/2)),  # xmin
           y_center-h/2,  # ymin
           min(x_center+w/2+right*alph*(w/2), im.size[0]-1),  # xmax
           y_center+h/2)  # ymax
 
    newW = box[2]-box[0]
    newH = box[3]-box[1]
    tmpImg = im.rotate(degree, center=(x_center, y_center)).crop(box)
    return tmpImg, newW, newH
