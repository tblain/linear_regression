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
        print(self.weights)

    def preactiv(self, x):
        return np.dot(self.weights, x) + self.bias

    def activ(self, res):
        return res
        #return max(0, res)

    def predict(self, x):
        return self.activ(self.preactiv(x))

    def predict_on_dataset(self, x):
        res = np.zeros(len(x))
        for i in range(len(x)):
            res[i] = self.predict(x.iloc[i])
        return res

    def loss(self, y, res): # 1 example
        #print("====")
        #print(y-res)
        #print(res)
        return (res - y)**2
        # return ((y - res) ** 2) / 2

    def activation_deriv(self, res):
        if res > 0:
            return 1
        else:
            return 0

    def cost(self, res, y): # dataset
        total_cost = 0
        for p, t in zip(res, y):
            print(p)
            total_cost += np.dot(p, t)
        return total_cost / res.shape[0]

    def fit(self, x, y, step, epochs, batch_size, learning_rate, validation_datas):
        x_val = validation_datas[0]
        y_val = validation_datas[1]
        x = x
        y = y
        i = 0
        epo = 0

        for etape in range(step):
            i += 1
            bx = x.iloc[batch_size*i : batch_size*(i+1) - 1]
            by = y.iloc[batch_size*i : batch_size*(i+1) - 1]

            w_gradient = np.zeros(self.weights.shape)
            b_gradient = 0
            print("cost: ", self.cost(x, y))

            for k in range(len(bx)):
                pre_a = self.preactiv(bx.iloc[k])
                post_a = self.activ(pre_a)
                #print("============")
                #print(post_a)
                #print(bx.iloc[k])
                #print(self.weights)
                loss = self.loss(by.iloc[k], post_a)
                act_deriv = 1 #self.activation_deriv(post_a)
                w_gradient += loss * act_deriv * bx.iloc[k]
                #print(w_gradient)
                b_gradient += loss * act_deriv

            self.weights -= learning_rate * (1 / batch_size) * w_gradient
            self.bias    -= learning_rate * (1 / batch_size) * b_gradient

            if x.shape[0] < batch_size*(i+1):
                print(" permute: ", x.shape[0])
                p = np.random.permutation(len(x))
                x = x.iloc[p]
                y = y.iloc[p]
                i = 0

            if etape != 0 and etape % (step / epochs) == 0:
                print("=====================")
            #if True:
                #print(i * batch_size)
                predic = self.predict_on_dataset(x_val)
                #for h, y in zip(predic, y_val):
                    #pass
                    #print("  ", h, " /  ", y)
                print("Epoch: ", epo, "     loss: ", ((predic - y_val) **2).mean())
                #print("predict: ", (predic - y_val).describe())
                print("weights: ", self.weights)
                print("bias: ", self.bias)
                epo += 1


dataset = pd.read_csv("./winequality-red.csv")

train = dataset
#validation = dataset.tail(199)

x_train = train.drop('quality', 1)
y_train = train.quality

# train_sata = pd.read_csv("./train.csv")
# x_train = train_sata.drop('target', 1)
# y_train = train_sata.target

cali_dataframe = pd.read_csv("./california_housing_train.csv")
x_train = cali_dataframe.drop("median_house_value", 1)
y_train = cali_dataframe.median_house_value / 1000

def normalize(x):
    return (x-min(x))/(max(x)-min(x))

for x in x_train:
    #x_train[x] = normalize(x_train[x])
    pass


x_val = x_train.tail(500)
y_val = y_train.tail(500)

model = Model_regression(x_train.shape[1])

model.fit(x_train, y_train, step=1000, epochs=10, batch_size=20, learning_rate=0.00000000000005, validation_datas=(x_val, y_val))
