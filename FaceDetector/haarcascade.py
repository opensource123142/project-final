# -*- coding: utf-8 -*-
"""
Created on Wed Nov 4 
@author: jongwon Kim 
         Deep.I Inc.
"""


import cv2
import timeit

# 영상 검출기
def videoDetector(cam,cascade):
    while True:
        start_t = timeit.default_timer()
        """ 알고리즘 연산 시작"""
        ret,img = cam.read()
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        results = cascade.detectMultiScale(gray,          
                                           scaleFactor= 1.1,
                                           minNeighbors=5,  
                                           minSize=(20,20) 
                                           )
                                                                           
        for box in results:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
     
        """ 알고리즘 연산종료 """ 
        terminate_t = timeit.default_timer()
        FPS = 'fps' + str(int(1./(terminate_t - start_t )))
        cv2.putText(img,FPS,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        cv2.imshow('facenet',img)
        if cv2.waitKey(1) > 0: 
            break

# 사진 검출기   
def imgDetector(img,cascade):

    img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    results = cascade.detectMultiScale(gray,            
                                       scaleFactor= 1.5,
                                       minNeighbors=5,  
                                       minSize=(20,20)  
                                       )        
    for box in results:
        x, y, w, h = box
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)   
    cv2.imshow('facenet',img)  
    cv2.waitKey(10000)

    


# 가중치 파일 경로
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

# 영상 파일 
cam = cv2.VideoCapture('test.mp4')
# 이미지 파일
img = cv2.imread('sample.jpg')

# 영상 탐지기
videoDetector(cam,cascade)
# 사진 탐지기
#imgDetector(img, cascade)