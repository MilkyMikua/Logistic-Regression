# Build a logistic regression model
# Each training example can contain 1 feature.
# Created Oct 1 2023, Last modified:
# Yurui Huang

# By using a new dataSet, you may consider modify:
# Pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FeatureScaling

path = r'C:\Users\h8611\OneDrive\Desktop\MLS Slides\HousePrice\diabetes.csv'  # Location of the dataset and read it
dataset = pd.read_csv(path)


x_train = np.array(dataset.iloc[:, 0:-1])   # consider the first 6 columns as features of a single training example
y_train = np.array(dataset.iloc[:, -1])  # The last column is the target
x_trainOriginal = x_train  # Save for rescale
mean = np.mean(x_train)  # Save for rescale
diff = np.max(x_train) - np.min(x_train)  # Save for rescale

x_train = FeatureScaling.meanNormal(x_train)  # rescaling features. Rescaling data improves data processing. Notice: in this case, MSE does not converage without rescaling

m = x_train.shape[0]  # amount of training examples. row
n = x_train.shape[1]  # number of features. Column

w= np.array([0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5])  # initial slope, Make sure
b = 0.02  # initial bias
a = 0.05  # initial learning rate

epoch = 100 + int(m/100)  # Set expected iteration

def sigmoid(w, b, x):  # The logistic regression model. Output a number between 0 and 1. which is a probability of a sample being a positive prediction.
    """
    The logstic Regression model is 1 / {1 + exp[-(W*x + b)] }
    def the f_wb = W*x + b, where w and x are vector, perform a dot product and get a scalar
    :param w: weight
    :param b: bias
    :param x: Training example
    :return:
    """
    # for the threshold, apply decision boundry. Refer to Course1_wk3_pg12
    f_wb = np.dot(w, x) + b
    g_z = 1 / (1 + np.exp(-f_wb))
    #y_hat = (g_z > 0.4)  # Set threshold at 40%. We set threshold at a probability(40%) since we dont want to miss a positive case.
    return g_z

def lossFun(w, b, x, y, m):  # Compute MSE
    '''
    :param w: slope
    :param b: bias
    :param x: training example feature
    :param y: actual output
    :param m: number of training examples
    :return: mean square error
    '''
    sum = 0
    for i in range(m):
        f_wb = sigmoid(w, b, x[i])
        y1 = -y[i] * np.log(f_wb)  # at y_i = 1
        y0 = (1-y[i]) * np.log(f_wb)  # at y_i = 0
        MSE = y1 - y0
        sum = sum + MSE

    MSE = sum / m
    return MSE

def grad_descent(w, b, a, x, y):
    """
    :param w: weight w
    :param b: bias b
    :param a: learning rate alpha
    :param x: training example's features' values
    :param y: training examples' actual value
    :return: update parameter w and b with gradient descent
    I tried to not use any loop
    """

    y_pred = np.dot(x, w) + b  # perform dot product of W and X, then + b
    diff = y_pred - y  # calculate the difference between y_prediction and y_actual

    # update parameter w
    w_i = x * diff[:, np.newaxis]  # broadcasting. To update parameter w need compute diff * x_train
    sum_w = np.sum(w_i, axis=1)  # sum each row of w_i. return a vector
    mean_w = np.mean(sum_w)  # Find the mean of the vector.
    learnRate_w = a * mean_w  # scale with learning rate
    w = w - learnRate_w

    # update parameter b
    b_i = diff
    mean_b = np.mean(b_i)
    learnRate_b = a * mean_b
    b = b - learnRate_b

    return w, b

# The rescle is used to scalize any new x samples into the same scale as the training examples.
# This means, we can not use Sigmod to process any new x sample, since direct processing a not-yet-scalized x sample is not precise
# If we have a new x sample, we need to use rescale function to predict the probablity, since rescale uses the exactly same idea as sigmoid but scale the new sample x
def rescale(x, w, b, mean, diff):
    x = (x-mean) / diff  # re-scale the input
    # x = x*x + x  # feature engining
    f_wb = np.dot(w, x) + b  # perform dot product to get a scalar. Here, w and x can be a vector
    return f_wb


# Start iterating to train the model ( Update parameter w and b with gradient descent)
# Also, record each iteration's MSE
MSE_record = []
for i in range(epoch):
    w, b = grad_descent(w, b, a, x_train, y_train)  # Update w and b with Grad_descent
    MSE_record.append(lossFun(w, b, x_train, y_train, m))  # record this iteration's MSE


print(w, "Optimized w")
print(b, "Optimized b")
plt.plot(MSE_record)  # I checked with a dataset, MSE do decrease. If did not converge, some thing from dataset/learning rate/Feature scaling etc might be wrong
plt.show()



# I want to use the optimized w and b to predict the original dataset
# However, the precision is not very correct
# I need to check True positive and False positive rate
all_pred = []
count = 0
for i in range(0, len(x_trainOriginal)):
    x_sample = x_trainOriginal[i]
    #print(x_sample, str(i) + "th x sample")
    pred = rescale(x_sample, w, b, mean, diff)
    if pred > 0.4:  # Set the threshold to 40%. Which means model evalute a sample's feature, and claim a sample is diabite if model output a number greater than 40%
        print(x_sample, "x_sample")
        print(pred, "Prediction")
        count = count + 1  # Count total postive prediction.
    all_pred.append(pred)
    #print(pred, str(i) + "th pred")
    #print(y_train[i], "the actual target \n")

print(count, " of predicted true sample")





