from rhizoscan import get_data_path
from matplotlib import pyplot as plt

from skimage import  io
from skimage import filters
import skimage.measure
import numpy as np
import cv2
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from sklearn.cluster import KMeans





image = io.imread('big.jpg')


################### crop image #################################"

 #image[hbound[1]:,:] = 0, En bas horizontale

image[430:,:] = 0 

# image[:hbound[0], :] = 0, En haut horizontalement

image[:39,:] = 0

# image[: , :wbound[0] ] = 0, a gauche verticale

image[: , :40] = 0

# image[: , wbound[1]:] = 0 a droite verticalement
image[: , 450:] = 0 


for i in range(5):
      image = cv2.morphologyEx(image,cv2.MORPH_OPEN,(25,25))
     
     ###""" Segmentation-Thresholding """###

     #cv2.adaptivethreshold(src,adaptiveMethod,thresholType,blocksize,c[,dst])



bin_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                  cv2.THRESH_BINARY_INV,101,25)

bin_image[430:,:] = 0 
bin_image[:39, :] =  0
bin_image[:, :40] = 0
bin_image[:, 450:] = 0 


                            
                            
################### Kept biggest connected component ################

label_image = skimage.measure.label(bin_image)
regions     = skimage.measure.regionprops(label_image)

###certaintly the best methods is to kept the area superior to a x define
### value

region = max(regions,key = lambda x: x.area)

im_seg = label_image.copy()

im_seg[im_seg != region.label] = 0

im_seg[im_seg > 0 ] = 255 

im_seg[430: ,:] = 0 
im_seg[:39 , :] = 0
im_seg[: , :40] = 0
im_seg[: ,450:] = 0 

####################detect leaves and seed #################################

# noise removal
kernel = np.ones((2,2),np.uint8)
#for i in range(5):

opening = cv2.morphologyEx(im_seg.astype(np.uint8),cv2.MORPH_OPEN, kernel, iterations = 4)
    
opening[430: ,:] = 0 
opening[:39 , :] = 0
opening[: , :40] = 0
opening[: ,450:] = 0 

############  Kmeans on im_seg ###################################

#kmeans = KMeans(n_clusters=5, random_state=0).fit(opening.astype(np.float64))


####################" Plot #######################################

#plt.figure(figsize = (60, 60))
#plt.subplot(151)
#~ plt.figure(0)
#plt.imshow(image)
#~ plt.axis('off')
#plt.subplot(152)
#~ plt.figure(1)
#plt.imshow(bin_image,cmap='gray', interpolation='bilinear')
#~ plt.axis('off')


#plt.subplot(153)
plt.figure(2)
plt.imshow(im_seg,cmap='gray', interpolation='bilinear')
#~ plt.axis('off')

#plt.subplot(154) 
plt.figure(3)
plt.imshow(opening,cmap='gray', interpolation='bilinear')

plt.figure(4)
plt.imshow(kmeans,cmap='gray', interpolation='bilinear')
#~ plt.axis('off')
#~ plt.show()


#~ plt.tight_layout()
#~ plt.show()
