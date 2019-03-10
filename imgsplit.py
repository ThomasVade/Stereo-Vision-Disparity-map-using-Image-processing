import numpy as np
import cv2
import os
import glob
##from skimage.morphology import erosion, dilation, opening, closing, white_tophat,disk
##from skimage import morphology
##from skimage.morphology import reconstruction
##from skimage.exposure import rescale_intensity
#import main

map1_x=np.loadtxt('map1_x.out',dtype='float32',delimiter=',')
map1_y=np.loadtxt('map1_y.out',dtype='float32',delimiter=',')
map2_x=np.loadtxt('map2_x.out',dtype='float32',delimiter=',')
map2_y=np.loadtxt('map2_y.out',dtype='float32',delimiter=',')
filecounter=0
count=0
##main_ob = main.counter()

w=0
while(w<=1):
    cap=cv2.VideoCapture(1)
    w+=1
    ret, img = cap.read()
    cv2.imshow("input", img)
    cv2.imwrite("test.bmp",img) # writes image test.bmp to disk
    



    # while True:
    #     ret, img = cap.read()
    #     cv2.imshow("input", img)
    #     cv2.imwrite("test.bmp",img) # writes image test.bmp to disk

    # s, im = cap.read() # captures image
    # cv2.imshow("Test Picture", im) # displays captured image
    # cv2.imwrite("test.bmp",im) # writes image test.bmp to disk

    fgbg = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    if ret==True:
        img=frame.copy()
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img=fgbg.apply(img)
        h,w=img.shape
        im_left=img[:,:int(w/2)]
        im_right=img[:,int(w/2):]

        # write the flipped frame
        im_left_remapped=cv2.remap(im_left,map1_x,map1_y,cv2.INTER_CUBIC)
        im_right_remapped=cv2.remap(im_right,map2_x,map2_y,cv2.INTER_CUBIC)

        out=np.hstack((im_left_remapped,im_right_remapped))
    ##        cv2.imshow('Remapped/remappedL',im_left_remapped)
    ##        cv2.imshow('Remapped/remappedR',im_right_remapped)
        count += 1
    ##        for i in range(0,out.shape[0],30):
    ##            cv2.line(out,(0,i),(out.shape[1],i),(0,255,255),3)
    ##        cv2.imshow('out',out)
        window_size = 5                  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,#0
            numDisparities=16,    #16     # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=9, #9
            P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=5, #1
            uniquenessRatio=0, #0
            speckleWindowSize=5, #0
            speckleRange=2, #2
            preFilterCap=63, #63 #20
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # FILTER Parameters
        lmbda = 80000 #80000
        sigma = 1.2#1.2
        visual_multiplier = 1.0 #1.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)


        displ = left_matcher.compute(im_left_remapped, im_right_remapped)  # .astype(np.float32)/16
        #cv2.imshow('dis',displ)
        dispr = right_matcher.compute(im_right_remapped, im_left_remapped)  # .astype(np.float32)/16
        #cv2.imshow('disr',dispr)
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, im_left_remapped, None, dispr)  # important to put "imgL" here!!!

    ##        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        
        cv2.imshow('Disparity Map', filteredImg)
        cv2.imwrite("C:\\Users\\N!kh!l\\Downloads\\stereocameraprograme\\"+str(count)+".png", filteredImg)
    ##        main_ob.counter(disparitybm, frame[:,:320])

        if cv2.waitKey(5) & 0XFF==ord('q'):
            break
        else:
            break
cap.release()
cv2.destroyAllWindows()

