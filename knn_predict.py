import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from skimage import filters
from skimage import morphology
import skimage
from skimage import io
from skimage import transform, util
import cv2
from collections import Counter
class Predict:
    def __init__(self):
        self.train=self.prepare_data_train()    
        self.class_names=['krzyżyk', 'kółko', 'siatka']
    def read_image(self,path):
        img=skimage.io.imread(path,as_gray=True)
        img=skimage.filters.gaussian(img, sigma=1)
        img=skimage.morphology.erosion(img, selem=np.ones((8,8)) )
        img=skimage.transform.resize(img,(28,28))
        img=skimage.img_as_ubyte(img)
        return img
    def read_images_X(self):
        p=Path('./dataset/iksy')
        paths=[x for x in p.iterdir() if x.is_dir]
        x_data=[]
        for path in paths:
            x_data.append(self.read_image(path))
        return np.array(x_data)
    def read_images_o(self):
        p=Path('./dataset/kolka')
        paths=[x for x in p.iterdir() if x.is_dir]
        o_data=[]
        for path in paths:
            o_data.append(self.read_image(path))
        return np.array(o_data)
    def read_images_g(self):
        p=Path('./dataset/plansze')
        paths=[x for x in p.iterdir() if x.is_dir]
        g_data=[]
        for path in paths:
            g_data.append(self.read_image(path))
        return np.array(g_data)    
    def prepare_data_train(self):
        x_data=self.read_images_X()
        o_data=self.read_images_o()
        g_data=self.read_images_g()
        X=np.concatenate((x_data,o_data,g_data), axis=0)
        y=[]
        for i in range(len(x_data)):
            y.append(0)
        for i in range(len(o_data)):
            y.append(1)
        for i in range(len(g_data)):
            y.append(2)
        y=np.array(y)
        trainX=[xxx.reshape(28*28) for xxx in X]
        train = np.insert(trainX, 784, y, axis = 1)
        return train 
    def Euclidean_distance(self,row1, row2):
        distance = 0
        for i in range(len(row1)-1):
            distance += (int(row1[i])-int(row2[i]))**2
        return math.sqrt(distance)
    def Get_Neighbors(self,train, test_row, num):
        distance = list() 
        data = []
        for i,j in enumerate(train):
            dist = self.Euclidean_distance(test_row, j)
            distance.append(dist)
            data.append(j)
        distance = np.array(distance)
        data = np.array(data)
        index_dist = distance.argsort()
        data = data[index_dist]
        neighbors = data[:num]
        return neighbors    
    def predict_classification(self,img):
        test_row=self.convert_image(img)
        test_row=np.reshape(test_row,(28*28))
        num=15
        train=self.train
        Neighbors = self.Get_Neighbors(train, test_row, num)
        np.set_printoptions(threshold=np.inf)
        Classes = []
        for i in Neighbors:
            Classes.append(i[-1])
        prediction = max(Classes, key= Classes.count)
        return self.class_names[prediction]
    def accuracy(self,y_true, y_pred):
        n_correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                n_correct += 1
        acc = n_correct/len(y_true)
        return acc
    def convert_image(self,img):
        img=skimage.color.rgb2gray(img) 
        img=skimage.filters.gaussian(img, sigma=1)
        img=skimage.morphology.erosion(img, selem=np.ones((8,8)) )
        img=skimage.transform.resize(img,(28,28))
        img=skimage.img_as_ubyte(img)
        return img
