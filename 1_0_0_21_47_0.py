from PIL import Image
import cv2
import numpy as np
import os
## 断路器状态识别 以红色为基准 top识别开合闸状态 bottom识别(未)储能状态
def image_detection(src,info_sorted):
    max=10000
    ## 选出红色正方形
    for k,v in info_sorted:
        x,y,w,h=v
        im=src[y:y+h,x:x+w]
        b, g, r = cv2.split(im)
        if(np.mean(r)>np.mean(g) and np.mean(r)>np.mean(b) and abs(1-abs(w/h))<max):
            max=abs(1-abs(w/h))
            red_x,red_y,red_w,red_h=v
    top_x,top_y,top_w,top_h=[red_x,int(red_y-0.797*red_h),red_w,int(0.339*red_h)]
    bottom_x,bottom_y,bottom_w,bottom_h=[red_x,int(red_y+3.73*red_h),red_w,int(0.339*red_h)]
    im_top=src[top_y:top_y+top_h,top_x:top_x+top_w]
    im_bottom=src[bottom_y:bottom_y+bottom_h,bottom_x:bottom_x+bottom_w]
    top_b, top_g, top_r = cv2.split(im_top)
    bottom_b,bottom_g,bottom_r=cv2.split(im_bottom)
    state=[0,0]
    if(np.mean(top_r)>np.mean(top_b) and np.mean(top_r)>np.mean(top_g)):
        # print('top:红色')
        state[0]=1
    elif(np.mean(top_g)>np.mean(top_r) and np.mean(top_g)>np.mean(top_b)):
        # print('top:绿色')
        state[0]=0
    if(np.mean(bottom_b)/(np.mean(bottom_b)+np.mean(bottom_g)+np.mean(bottom_r))>=0.3):
        # print('bottom:白色')
        state[1]=0
    else:
        # print('bottom:黄色')
        state[1]=1
    # cv2.imshow("top", im_top)
    # cv2.imshow("bottom",im_bottom)
    # cv2.waitKey(0)
    print(state)
    return state
 # 提取图中轮廓的个数
def ShapeDetection(path):
    contours,hierarchy = cv2.findContours(path,cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_TC89_L1)  #寻找轮廓点
    # print(len(contours))
    info=dict()
    for obj in contours:
        area = cv2.contourArea(obj)  # 计算轮廓内区域的面积
        # cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  # 绘制轮廓线
        perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
        CornerNum = len(approx)  # 轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度
        info[area]=[x,y,w,h]
    info_sorted=sorted(info.items(),reverse=True)
    return info_sorted[:3]
"""
提取图中的颜色部分
"""
def ColorDetection(img,color_hsv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv=color_hsv[0]
    high_hsv=color_hsv[1]
    low_hsv = np.array(low_hsv)
    high_hsv = np.array(high_hsv)
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    return mask
def detection_21_47(src,red_hsv=[[0,83,55],[180,255,255]]):
    mask = ColorDetection(src, red_hsv)
    info_sorted = ShapeDetection(mask)
    state=image_detection(src, info_sorted)
    return state

if __name__=='__main__':
    rootDir='1_0_0_21_47_0'
    red_hsv = [[0, 83, 55], [180, 255, 255]]
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        # print(pathname)
        src = cv2.imread(pathname)
        print(detection_21_47(src, red_hsv))

