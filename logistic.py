from logistic import *
import numpy as np
from datasets import tr_x, tr_y, val_x, val_y
from util import sigmoid
import matplotlib.pyplot as plt

class Logistic:
    def __init__(self, maxIter, eta, lambda_ = None):
        self.maxIter = maxIter
        self.eta = eta
        self.lambda_ = lambda_
        self.w_vec = None

    # tr:val = 4:1 for report
    def train_GA(self, tr_x, tr_y):
        self.w_vec = np.zeros(2)
        for t in range(self.maxIter):
            temp_w = np.zeros(2) #temp_w: temp weig
            for j in range(len(tr_x)):  #add each instances(so that compute simga)
                temp_w = temp_w + np.multiply(tr_x[j], tr_y[j] - sigmoid(np.dot(self.w_vec, tr_x[j])))#이전 iteration에서의 weight를 가지고 update해 나가기
            self.w_vec = self.w_vec + self.eta * temp_w

    #tr:val = 4:1 for report
    def train_SGA(self, tr_x, tr_y):
        self.w_vec = np.zeros(2)
        for t in range(self.maxIter): #각 iteration t 마다 batch의 크기를 random하게 정하고, batch를 random하게 만들기->temp_w를 계산하기 ->w_vec 업데이트

            batch_size = np.random.randint(1, len(tr_x))
            batch_x = []*batch_size
            batch_y = []*batch_size
            for _ in range(batch_size): #batch 만들기
                idx = np.random.randint(len(tr_x))
                batch_x.append(tr_x[idx])
                batch_y.append(tr_y[idx])
            temp_w = np.zeros(2) #위에서 만들어진 batch를 가지고 업데이트 #temp_w: temp weight
            for j in range(batch_size):  #만들어진 batch를 바탕으로 temp_w를 계산하기#add each instances(so that compute simga)
                temp_w = temp_w + np.multiply(batch_x[j], batch_y[j] - sigmoid(np.dot(self.w_vec, batch_x[j]))) #이전 iteration에서의 weight를 가지고 update해 나가기
            self.w_vec = self.w_vec + self.eta * temp_w


    # tr:val = 7:3
    def train_reg_SGA(self, tr_x, tr_y):
        self.w_vec = np.zeros(2)
        for t in range(self.maxIter):
            batch_size = np.random.randint(len(tr_x))
            batch_x = [] * batch_size
            batch_y = [] * batch_size
            for _ in range(batch_size):  # batch에 element 넣기
                idx = np.random.randint(len(tr_x))
                batch_x.append(tr_x[idx])
                batch_y.append(tr_y[idx])

            temp_w = (-1) * (self.lambda_) * self.w_vec
            for j in range(batch_size):
                temp_w = temp_w + np.multiply(batch_x[j], batch_y[j] - sigmoid(np.dot(self.w_vec, batch_x[j])))
            self.w_vec = self.w_vec + self.eta * temp_w
    def predict(self, val_x):
        result = np.zeros(len(val_x))
        for i in range(len(val_x)):
            if sigmoid(np.dot(val_x[i], self.w_vec))>0.5: result[i]=1
        return result

def computeClassificationAcc(val_y, y_hat):
    count = 0
    for i in range(len(val_y)):
        if val_y[i] == y_hat[i]:
            count += 1
    return (count/len(val_y))


from logistic import *
model = Logistic(100, 0.01, lambda_=3) # set maximum iteration and eta (learning rate) for model updates
model.train_reg_SGA(tr_x, tr_y)
y_hat = model.predict(val_x)
acc = computeClassificationAcc(val_y, y_hat)
print(acc)