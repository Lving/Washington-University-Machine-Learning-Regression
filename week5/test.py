<<<<<<< HEAD
# encoding: utf-8
import graphlab
import numpy as np # note this allows us to refer to numpy as np instead
sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string,
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int)

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return feature_matrix, output_array

def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return predictions

X = np.array([[3.,5.,8.],[4.,12.,15.]])
print X

norms = np.linalg.norm(X, axis=0) # gives [norm(X[:,0]), norm(X[:,1]), norm(X[:,2])]
print norms

print X / norms
# gives [X[:,0]/norm(X[:,0]), X[:,1]/norm(X[:,1]), X[:,2]/norm(X[:,2])]
def normalize_features(features_matrix):
    norms = np.linalg.norm(features_matrix, axis=0)
    normalized_features = features_matrix / norms
    return normalized_features, norms

features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print norms
# should print
# [5.  10.  15.]

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)

simple_feature_matrix, norms = normalize_features(simple_feature_matrix)
weights = np.array([1., 4., 1.])
prediction = predict_output(simple_feature_matrix, weights)
# ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].
ro = np.zeros(weights.shape)
for i in range(len(weights)):
    ro[i] = np.dot(simple_feature_matrix[:,i], (output - prediction + weights[i] * simple_feature_matrix[:,i]))


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.dot(feature_matrix[:, i], (output - prediction + weights[i] * feature_matrix[:, i]))

    #     ```
    #        ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
    #     w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
    #        └ (ro[i] - lambda/2)     if ro[i] > lambda/2
    #     ```
    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2
    else:
        new_weight_i = 0.

    return new_weight_i

# should print 0.425558846691
import math
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]),
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    # 定义一个change 用来判断停止
    max_change = 1e10
    # 当max_change > tolerance时 Loop
    while max_change > tolerance:
        weights = initial_weights
        old_weights = np.zeros_like(weights)
        # 利用for循环来更新weight，
        for i in range(len(weights)):
            old_weights[i] = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
        max_change = abs((weights - old_weights).max())
    return weights

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)

weights
=======
# __author__ = "Administrator"
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.as_matrix()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.as_matrix()
    return feature_matrix, output_array


def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return predictions


def normalize_features(features_matrix):
    norms = np.linalg.norm(features_matrix, axis=0)
    normalized_features = features_matrix / norms
    return normalized_features, norms


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.dot(feature_matrix[:, i], (output - prediction + weights[i] * feature_matrix[:, i]))
    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2
    else:
        new_weight_i = 0.
    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    # 定义一个change 用来判断停止
    max_change = tolerance + 1
    # 当我将这个weights的初始化写在while循环的内部时，函数输出正确的值，但是在之后的调用的时候，
    # 仅返回最后一个的调用值。
    # 这个初始化，只需要完成一次即可，写在while循环中的时候， 就会初始化很多次
    weights = np.array(initial_weights)  # 只需要初始化一次
    # 当max_change > tolerance时 Loop
    old_weights = np.zeros(weights.shape)
    while max_change > tolerance:
        # 初始化old_weights 的值， 同来存放为weights的值， 对weights的循环之前赋值一次，
        # 利用for循环来更新weight，
        for i in range(len(weights)):
            old_weights[i] = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            # 循环完成， weights获得新的值，进行while判断，
        max_change = abs((weights - old_weights).max())
    return weights


train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated']

(train_feature_matrix, train_output) = get_numpy_data(train_data, all_features, 'price')
(normalize_train_feature_matrix, train_norms) = normalize_features(train_feature_matrix)

tmp_initialize_weights = np.zeros(train_feature_matrix[1].shape)
weights1e7 = lasso_cyclical_coordinate_descent(normalize_train_feature_matrix, train_output,
                                              tmp_initialize_weights, 1e7, 1)
weights1e4 = lasso_cyclical_coordinate_descent(normalize_train_feature_matrix, train_output,
                                              tmp_initialize_weights, 1e4, 5e5)
weights1e8 = lasso_cyclical_coordinate_descent(normalize_train_feature_matrix, train_output,
                                               tmp_initialize_weights, 1e8, 1)


normalized_weights1e4 = weights1e4 / train_norms
normalized_weights1e7 = weights1e7 / train_norms
normalized_weights1e8 = weights1e8 / train_norms

print normalized_weights1e7[3]


(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')
rss1e4 = ((predict_output(test_feature_matrix, normalized_weights1e4) - test_output)**2).sum()
print rss1e4

rss1e7 = ((predict_output(test_feature_matrix, normalized_weights1e7) - test_output)**2).sum()
print rss1e7

rss1e8 = ((predict_output(test_feature_matrix, normalized_weights1e8) - test_output)**2).sum()
print rss1e8
>>>>>>> 33e74921da9fc028f0052e957a08c8b9011154f8
