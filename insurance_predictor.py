import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model

df = pd.read_csv("insurance_data.csv")

# Returns first n columns
df.head()

# prints the csv file contents
# print(df)


# print(df.shape) -- (28,3)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    df[['age', 'affordibility']], df.bought_insurance, test_size=0.2, random_state=25)

print(Y_test)
print(Y_train)
# print(len(X_train))

# scaling the age between 0 and 1
X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100

# print(X_train_scaled)

# Note: Kernel Initializer is W and Bias Initializer is B
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid',
    kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train_scaled, Y_train, epochs=5000
)

model.save("/Users/ahankamat/Desktop/Deep Learning/insurance_data_model.keras")
model = load_model(
    '/Users/ahankamat/Desktop/Deep Learning/insurance_data_model.keras')
model.evaluate(X_test_scaled, Y_test)

print(X_test_scaled)
model.predict(X_test_scaled)

print(Y_test)


coef, intercept = model.get_weights()
print(coef, intercept)

# [5.060863] - w1
#  [1.4086521] - w2
# [-2.913703] - b



#  ----------------------------IMPLEMENTATION STARTS HERE-------------------------


# def sigmoid(x):
#     return 1/(1+math.exp(-x))

# def prediction_function(age , affordibility):
#     weighted_sum = coef[0] * age + coef[1] * affordibility + intercept

# prediction_function(.47 , 1)
# prediction_function(.18 , 1)


def log_loss(Y_true, Y_predicted):
    epsilon = 1e-15
    Y_predicted_new = [max(i, epsilon) for i in Y_predicted]
    Y_predicted_new = [min(i, 1-epsilon) for i in Y_predicted_new]
    Y_predicted_new = np.array(Y_predicted_new)
    return -np.mean(Y_true*np.log(Y_predicted_new)+(1-Y_true)*np.log(1-Y_predicted_new))

def sigmoid_numpy(x):
    return 1/(1+np.exp(-x))


class myNN:
    def __init__(self):
        self.w1=1
        self.w2=1
        self.bias=0
    
    def fit(self, X, Y, epochs, loss_threshold):
      self.w1, self.w2, self.bias = self.gradient_descent(
        X['age'], X['affordibility'], Y, epochs, loss_threshold)
    
    
    def predict(self , X_test):
        weighted_sum = self.w1 * X_test["age"] + self.w2 * X_test["affordibility"]+ self.bias
        return sigmoid_numpy(weighted_sum)
    
    
    def gradient_descent(self , age, affordibility, Y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordibility + bias
            Y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(Y_true, Y_predicted)

            w1d = (1/n)*np.dot(np.transpose(age), (Y_predicted-Y_true))
            w2d = (1/n)*np.dot(np.transpose(affordibility), (Y_predicted-Y_true))

            bias_d = np.mean(Y_predicted-Y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d

            if i%50==0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

            if loss is not None and loss <= loss_thresold:
                    break

        return w1, w2, bias

customModel = myNN()
customModel.fit(X_train_scaled , Y_train , 400 , 0.46)

customModel.predict(X_test_scaled)

# ----------------------------IMPLEMENTATION ENDS HERE-------------------------
