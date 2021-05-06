import cv2,glob
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.getcwd()
path1 = path + "\\indian rock python"
os.chdir(path1)
os.mkdir("IRP")

images=glob.glob("*.jpg")

for image in images:
    img=cv2.imread(image,1)

    re=cv2.resize(img,(500,500))
    #cv2.imshow("Checking",re)
  
   
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,500,500)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
#    img1 = cv2.medianBlur(img,5)


##    th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
##            cv2.THRESH_BINARY,11,2)

    #plt.imshow(th2,'gray')
    #cv2.imshow("checking",img1)
    #cv2.imshow("checking",img)
    #plt.colorbar()
##    cv2.waitKey(500)
##    cv2.destroyAllWindows()
    os.chdir("IRP ")
    cv2.imwrite("resized_"+image,img)
    os.chdir(path1)

