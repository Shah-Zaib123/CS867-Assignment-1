import numpy as np
from scipy import signal
import cv2 as cv
from skimage.util import random_noise

def rgbExclusion(img,num):
    bgr_image = img.copy()   
    bgr_image[:,:,num] = 0 #empty specified channel 
    return bgr_image

def convolution_2d_image(img, krnl):
    # This function which takes an image and a kernel and returns the convolution of them
    krnl = np.flipud(np.fliplr(krnl))    # for Fliping the kernel
    output_img = np.zeros_like(img)            # convolution output
    img_pd = np.zeros((img.shape[0] + 2, img.shape[1] + 2))  # Adding zero padding to image 
    img_pd[1:-1, 1:-1] = img 
    # implementing convolution operation (element wise multiplication and summation), and storing the result in the output_img variable.
    for x in range(img.shape[0]):     # Loop over every pixel of the image
        for y in range(img.shape[1]):
            output_img[x,y]=(krnl*img_pd[x:x+3,y:y+3]).sum()  # element-wise multiplication and summation 
    return output_img

def box_filter(img):
        k = np.ones((5,5),np.float32)/25 #5x5 box filter kernel
        r_image = signal.convolve2d(img,k, 'same')
        return r_image
    
def guassian_noise(img):
        gauss = np.random.normal(0,1,img.size)
        gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        img_gauss = cv.add(img,gauss)
        return img_gauss
    
def sp_noise(img):
        noise_img = random_noise(img, mode='s&p',amount=0.1) # Add salt-and-pepper noise to the image. 
       # The above function returns a floating-point image
       # on the range [0, 1], thus we changed it to 'uint8' and from [0,255]
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        return noise_img

