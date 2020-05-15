import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image

def save(image):
    print("Do you wish to save the edited image? (press 1 for yes, 2 for no)")
    ans = input()
    if(ans == '1'):
        print("Enter file name:")
        name = input() + '.jpg'
        cv2.imwrite(name, image)
    else:
        exit

def grayscaling(image):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow('original', image)
    cv2.imshow('threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save(threshold)

def otsu_thresh(image):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(grayscaled, 125, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = np.invert(threshold) # invert colours so it displays similar to MNIST dataset numbers
    cv2.imshow('original', img)
    cv2.imshow('Otsu threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save(threshold)

def adaptive_thresh(image):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    threshold = np.invert(threshold)
    cv2.imshow('original', img)
    cv2.imshow('Adaptive threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save(threshold)
    


img = cv2.imread('button-29.jpg')

# grayscaling(img)
otsu_thresh(img)

# adaptive_thresh(img)

# cv2.imwrite('edited.jpg', threshold)


