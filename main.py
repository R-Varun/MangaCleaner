import cv2
import numpy as np

# img = cv2.imread('manga.jpg')
# #Display Image
# for i in range(100):
#     for j in range (100):
#         img[i,j] = [0,0,0]
#
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from matplotlib import pyplot as plt

# img = cv2.imread('manga.jpg',0)
# # edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()




image = cv2.imread("manga2.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) # threshold
thresh = cv2.Canny(gray, 100, 200)
#hopefully this would get rid of some noise as text is relatively dense
thresh = cv2.medianBlur(thresh, 1)
cv2.imshow('image',thresh)
cv2.waitKey(0)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 12) # dilate
s, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    [x,y,w,h] = cv2.boundingRect(contour)

    if h>200 and w>200:
        continue

    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    print("done")

# write original image with added contours to disk
cv2.imwrite("contoured.jpg", image)