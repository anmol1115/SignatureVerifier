import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

import numpy as np
import matplotlib.pyplot as plt

class CNN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5)

        self.fc1 = nn.Linear(in_features=10*6*6, out_features=180)
        self.fc2 = nn.Linear(in_features=180, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=2)

    def forward(self, t):
        t = self.conv1(t)
        t = f.relu(t)
        t = f.max_pool2d(t, kernel_size=4, stride=4)

        t = self.conv2(t)
        t = f.relu(t)
        t = f.max_pool2d(t, kernel_size=3, stride=3)

        t = self.conv3(t)
        t = f.relu(t)
        t = f.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 10*6*6)
        t = f.relu(self.fc1(t))

        t = f.relu(self.fc2(t))
        t = f.relu(self.fc3(t))

        t = self.out(t)
        return t

    def get_num_correct(self, labels, preds):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def graph(self, losses, correct):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Metrics')
        ax1.set_title('Losses')
        ax1.plot(losses)
        ax2.set_title('Total Correct')
        ax2.plot(correct)

class CNN_Classifier_Cedar(CNN_Classifier):
    def __init__(self):
        super().__init__()
        
    def forward(self, t):
        t = self.conv1(t)
        t = f.leaky_relu(t)
        t = f.max_pool2d(t, kernel_size=4, stride=4)

        t = self.conv2(t)
        t = f.leaky_relu(t)
        t = f.max_pool2d(t, kernel_size=3, stride=3)

        t = self.conv3(t)
        t = f.leaky_relu(t)
        t = f.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 10*6*6)
        t = f.leaky_relu(self.fc1(t))

        t = f.leaky_relu(self.fc2(t))
        t = f.leaky_relu(self.fc3(t))

        t = self.out(t)
        return t

class PCA_Transformation:
    def __init__(self):
        self.pca = PCA()
        self.recommended = 0
        
        self.classifier_names = ['SVC', 'NaiveBayes', 'KNN', 'Decision Tree',
        'Logistic Regression', 'Linear Discriminant Analysis']
        self.classifiers = [SVC(), GaussianNB(), KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(), LogisticRegression(), LinearDiscriminantAnalysis()]

    def findOptimumComponent(self, X):
        self.pca.fit(X)
        explained_variance = self.pca.explained_variance_

        plt.figure(1, figsize=(10,6))
        plt.plot(explained_variance, linewidth=2)
        plt.xlabel('Components')
        plt.ylabel('Explained Variaces')
        plt.show()

        self.recommend(explained_variance)

    def recommend(self, ndarray):
        for index in range(len(ndarray)-1):
            if ndarray[index]<0.5 and ndarray[index]-ndarray[index+1] < 0.01:
                print('Recomended value for n_component is {}'.format(index))
                self.recommended = index
                break
    
    def fit(self, X, n_component=None):
        if n_component == None:
            n_component = self.recommended
        self.pca = PCA(n_components = n_component)
        self.pca.fit(X)

    def transformData(self, X1, X2):
        return self.pca.transform(X1), self.pca.transform(X2)

    def classifyAndEvaluate(self, X_train, X_test, y_train, y_test):
        for index in range(len(self.classifiers)):
            self.classifiers[index].fit(X_train, y_train)
            y_pred = self.classifiers[index].predict(X_test)
            print('Model Name: ', self.classifier_names[index])
            print('Model Accuracy: ', accuracy_score(y_pred, y_test))
            print('Model Precision: ', precision_score(y_pred, y_test))
            print('Model f_score: ', f1_score(y_pred, y_test))
            print('Model Recall Score: ', recall_score(y_pred, y_test))
            print('\n')

def loadData(directory, toType='ndarray'):
    path = './Dataset/sign_data/'
    data = torch.tensor([])
    label = torch.tensor([])

    for folder in os.listdir(path+directory):
        for img_addr in os.listdir(path+directory+'/'+folder):
            img = cv2.imread(path+directory+'/'+folder+'/'+img_addr)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_tensor = transforms.ToTensor()(img)
            if toType == 'tensor':
                img_tensor = img_tensor.unsqueeze(0)

            data = torch.cat((data, img_tensor))
            if folder[-1] == 'g':
                label = torch.cat((label, torch.tensor([1])))
            else:
                label = torch.cat((label, torch.tensor([0])))

    data = data.numpy()
    label = label.numpy()
    data, label = shuffle(data, label)

    if toType == 'tensor':
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

    return data, label

def loadCedarData(toType='ndarray', test_size=0.25):
    path = './Dataset/signatures'
    data = torch.tensor([])
    label = torch.tensor([])

    for folder in os.listdir(path):
        for img_addr in os.listdir(path+'/'+folder):
            if img_addr == 'Thumbs.db':
                continue
            img = cv2.imread(path+'/'+folder+'/'+img_addr)
            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_tensor = transforms.ToTensor()(img)
            if toType == 'tensor':
                img_tensor = img_tensor.unsqueeze(0)

            data = torch.cat((data, img_tensor))
            if folder[5] == 'f':
                label = torch.cat((label, torch.tensor([1])))
            else:
                label = torch.cat((label, torch.tensor([0])))

    data = data.numpy()
    label = label.numpy()
    data, label = shuffle(data, label)

    if toType == 'tensor':
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        return data, label

    return train_test_split(data, label, test_size=test_size, random_state=0)