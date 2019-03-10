import numpy as np
import pandas as pd
from tqdm import tqdm
import math


class Model_regression:
    def __init__(self, input_shape):
        """
        """
        self.bias = 1
        self.weights = np.random.rand(input_shape)

    def preactiv(self, x):
        return np.dot(self.weights, x) + self.bias

    def activ(self, res):
        return res
        # return max(0, res)

    def predict(self, x):
        return self.activ(self.preactiv(x))

    def predict_on_dataset(self, x):
        res = np.zeros(len(x))
        for i in range(len(x)):
            res[i] = self.predict(x.iloc[i])
        return res

    def loss(self, res, y):
        return y - res
        # return ((y - res) ** 2) / 2

    def activation_deriv(self, res):
        if res > 0:
            return 1
        else:
            return 0

    def fit(self, x, y, step, batch_size, learning_rate, validation_datas):
        x_val = validation_datas[0]
        y_val = validation_datas[1]

        for i in tqdm(range(step)):
            bx = x.loc[batch_size*i:batch_size*(i+1) - 1]
            by = y.loc[batch_size*i:batch_size*(i+1) - 1]

            w_gradient = np.zeros(self.weights.shape)
            b_gradient = 0

            for k in range(len(bx)):
                pre_a = self.preactiv(bx.iloc[k])
                post_a = self.activ(pre_a)
                loss = self.loss(post_a, by.iloc[k])
                act_deriv = 1 #self.activation_deriv(post_a)
                w_gradient += loss * act_deriv * bx.iloc[k]
                b_gradient += loss * act_deriv

            self.weights -= learning_rate * w_gradient
            self.bias    -= learning_rate * b_gradient

            # if i != 0 and i % 10 == 0:
            if True:
                print("Step: ", i)
                # print(self.weights)
                predic = self.predict_on_dataset(x_val)
                print(((predic - y_val) **2).mean())


dataset = pd.read_csv("./winequality-red.csv")

train = dataset.head(1400)
validation = dataset.tail(199)

x_train = train.drop('quality', 1)
y_train = train.quality

# train_sata = pd.read_csv("./train.csv")
# x_train = train_sata.drop('target', 1)
# y_train = train_sata.target

cali_dataframe = pd.read_csv("./california_housing_train.csv")
x_train = cali_dataframe.drop("median_house_value", 1)
y_train = cali_dataframe.median_house_value

def normalize(x):
    return (x-min(x))/(max(x)-min(x))

for x in x_train:
    x_train[x] = normalize(x_train[x])

y_train = normalize(y_train)

x_val = x_train.tail(500)
y_val = y_train.tail(500)

print(x_train.describe())

model = Model_regression(x_train.shape[1])
model.fit(x_train, y_train, step=100, batch_size=10, learning_rate=0.0000001, validation_datas=(x_val, y_val))
