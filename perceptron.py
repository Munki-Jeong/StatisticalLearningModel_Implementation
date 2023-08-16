from perceptron import *
import numpy as np
from datasets import x, y, tr_x, tr_y, val_x, val_y
import matplotlib.pyplot as plt
from util import plot_decision_boundary_logistic_regression

from sklearn.linear_model import Perceptron as sk_Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split

tr_y = np.where(tr_y == 0, -1, tr_y)
val_y = np.where(val_y == 0, -1, val_y)

class Perceptron:
    def __init__(self, threshold, maxEpoch = 60):
        self.threshold = threshold
        self.maxEpoch= maxEpoch
        self.w_vec = np.zeros(2)
    def train(self, tr_x, tr_y):
        self.w_vec = np.zeros(2)
        for t in range(self.maxEpoch):
            for j in range(len(tr_x)):
                if np.dot(self.w_vec, tr_x[j]) > self.threshold:
                    y_hat1 = 1
                else: y_hat1 = -1

                if y_hat1 == tr_y[j]: pass #weight 그대로
                elif tr_y[j] == 1 :  self.w_vec = self.w_vec + np.array(tr_x[j])
                else: self.w_vec = self.w_vec - np.array(tr_x[j]) #tr_[y] == -1 이고, y_hat1 == 1인경우
    def predict(self, val_x):
        result = np.full(len(val_x), -1) #일단 -1로 채워둔 후,
        for j in range(len(val_x)):
            if np.dot(self.w_vec, val_x[j])>self.threshold:
                result[j]=1     #threshold를 넘기면 1로 바꾸기
        return result


def computeClassificationAcc(val_y, y_hat):
    count = 0
    for i in range(len(val_y)):
        if val_y[i] == y_hat[i]:
            count += 1
    return (count/len(val_y))

from perceptron import *
model = Perceptron(2) # set threshold value or
model.train(tr_x, tr_y)
y_hat = model.predict(val_x)
acc = computeClassificationAcc(val_y, y_hat)
print(acc)




model = Perceptron(2, maxEpoch=100)
sk_model = sk_Perceptron(max_iter=100)

model.train(tr_x, tr_y)
sk_model.fit(tr_x, tr_y)

model.predict(val_x)
sk_model.predict(val_x)

print(computeClassificationAcc(val_y, model.predict(val_x)))
print(computeClassificationAcc(val_y, sk_model.predict(val_x)))


def cross_val_score_custom_perceptron(model, X, y, cv=5, scoring='accuracy'):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        if scoring == 'accuracy':
            score = computeClassificationAcc(y_test, y_pred)
            scores.append(score)

    return scores


def learning_curve_custom(estimator, X, y, train_sizes, cv):
    train_scores = []
    val_scores = []

    for train_size in train_sizes:
        train_score_folds = []
        val_score_folds = []

        for i in range(cv):
            X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, stratify=y)
            estimator_copy = deepcopy(estimator)
            estimator_copy.train_GA(X_train, y_train)
            train_score_folds.append(computeClassificationAcc(y_train, estimator_copy.predict(X_train)))
            val_score_folds.append(computeClassificationAcc(y_val, estimator_copy.predict(X_val)))

        train_scores.append(train_score_folds)
        val_scores.append(val_score_folds)

    return train_sizes, train_scores, val_scores

my_score = cross_val_score_custom_perceptron(model, x, y, cv=5, scoring='accuracy')
sk_score = cross_val_score(sk_model, x, y, cv=5)
print(my_score, sk_score)

my_cv = [0.35, 0.1, 0.15, 0.0, 0.0]
skperceptron_cv = [0.5, 0.95, 1.,   1. ,  0.95]

x = np.arange(len(my_cv))
width = 0.3

plt.bar(x , my_cv, width, label='my_perceptron_model')
plt.bar(x + width, skperceptron_cv, width, label='scikit-perceptron')
plt.xlabel('Fold')
plt.ylabel('Cross-validation score')
plt.title('Cross-validation Score Comparison')
plt.xticks(x, [f'Fold {i+1}' for i in range(len(my_cv))])
plt.legend()
plt.show()
