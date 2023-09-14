import pandas as pd
import numpy as np
from csv import reader
import seaborn as sns
import matplotlib.pyplot as plt
import random
from random import randrange
from math import sqrt
from math import exp
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Importing Dataset , Wine dataset is located at D:\\Krishna_Python\\ML_Assignment\\wine.csv
wineDataSet = pd.read_csv("D:\\Krishna_Python\\ML_Assignment\\wine.csv")
wineDataSet


wineDataSet.head(10)
wineDataSet.info()
wineDataSet.describe().T
# EDA 
#Checking null values
wineDataSet.isnull().sum()
categorical = [var for var in wineDataSet.columns if wineDataSet[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)
# Encoding categorical variable
wineDataSet['quality_categorical'] = wineDataSet['quality'].astype(
    'category').cat.codes
wineDataSet.head()
wineDataSet.hist(bins=20, figsize=(10, 10))
plt.show()
plt.bar(wineDataSet['quality'], wineDataSet['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()
plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=0, bottom=0.5, right=0.9,
                    top=0.9, wspace=0.5, hspace=0.8)
plt.subplot(141)
plt.title('Percentage of good and bad quality wine', fontsize=20)
wineDataSet['quality'].value_counts().plot.pie(autopct="%1.1f%%")

# Dropping quality column
dframefinal = wineDataSet.drop('quality', axis=1)
dframefinal.info()

# Segregating dependent and independent variables
X = dframefinal.drop('quality_categorical', axis=1)
Y = dframefinal['quality_categorical']

"""(a) & (c) Implement Naive Bayes classifier algorithm from scratch with k-fold cross validation technique for two class classification."""

# Split the dataset by class values, returns a dictionary


def separateByClass(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Load a CSV file
def load_csvfile(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        i = 0
        for row in csv_reader:
            if not row:
                continue
            if i == 0:
                i = i+1
                continue
            dataset.append(row)
    return dataset
# Calculate the mean of a list of numbers


def mean(numbers):

    return sum(numbers)/float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)
# Calculate the mean, standard deviation and count for each column in the wine dataset


def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column))
                 for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
# Convert string column to float


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
# Convert string column to integer


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        if value == "good":
            lookup[value] = 1
        elif value == "bad":
            lookup[value] = 0
        print('[%s] => %d' % (value, lookup[value]))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
# Split dataset by class then calculate statistics for each row


def summarize_by_class(dataset):
    separated = separateByClass(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries
# Calculate the Gaussian probability distribution function for x


def calculate_Gprobability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent
# Calculate the probabilities of predicting each class for a given row


def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / \
            float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_Gprobability(
                row[i], mean, stdev)
    return probabilities
# Split a dataset into k folds for cross validation


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
# Evaluate an algorithm using a cross validation split


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    myindex = 0
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        print('Naive Bayes fold is %d and accuracy: %.3f' % (myindex, accuracy))
        myindex = myindex + 1
        scores.append(accuracy)
    return scores
# Predict the class for a given row


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label
# Naive Bayes Algorithm


def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)


# Make a prediction with Naive Bayes on Wine Dataset
# seed(1)
filename = 'D:\\Krishna_Python\\ML_Assignment\\wine.csv'
winedataset = load_csvfile(filename)
for i in range(len(winedataset[0])-1):
    str_column_to_float(winedataset, i)
# convert class column to integers
str_column_to_int(winedataset, len(winedataset[0])-1)
# evaluate algorithm
n_folds = 10
scores_nb = evaluate_algorithm(winedataset, naive_bayes, n_folds)
print('Naive Bayes Scores: %s' % scores_nb)
print('Naive Bayes Mean Accuracy: %.3f%%' % (sum(scores_nb)/float(len(scores_nb))))
# fit model
training_set = winedataset[0:1279]
test_set = winedataset[1279:1599]
model = summarize_by_class(training_set)

"""(d) Accuracy, Confusion matrix, ROC curve and F1 Score for Naive Bayes classifier algorithm"""

y_pred_for_naive_bayes = list()
y_test_for_naive_bayes = list()
for row in test_set:
    y_pred_for_naive_bayes.append(predict(model, row))
    y_test_for_naive_bayes.append(row[11])

# confusion matrix for Naive Bayes classifier algorithm
con_mat_nb = confusion_matrix(y_test_for_naive_bayes, y_pred_for_naive_bayes)
print('Naive Bayes Confussion Matrix')
print(con_mat_nb)
plt.figure(figsize=(10, 8))
sns.heatmap(con_mat_nb, fmt='.0f', annot=True)

y_prob_for_naive_bayes = []
# Class Prediction for a given row


def predictprob(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return probability


for row in test_set:
    y_prob_for_naive_bayes.append(predictprob(model, row))
fpr_nb, tpr_nb, _ = metrics.roc_curve(
    y_test_for_naive_bayes,  y_prob_for_naive_bayes)
# create ROC curve
plt.plot(fpr_nb, tpr_nb, marker='.', label='ROC curve for Naive Bayes')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()

# accuracy for naive bayes
Naive_Bayes_Accuracy_Score = metrics.accuracy_score(
    y_test_for_naive_bayes, y_pred_for_naive_bayes)
Naive_Bayes_Accuracy_Score

# Publishing f1 score for Naive Bayes classifier algorithm
f1ScoreForNB = f1_score(y_test_for_naive_bayes, y_pred_for_naive_bayes)
print('F1 score for Naive Bayes: %f' % f1ScoreForNB)

"""(b) Implement Logistic regression from scratch"""

# Creating training & test set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()
X_train_for_log_regr = sc.fit_transform(X_train.values)
X_test_for_log_regr = sc.transform(X_test.values)

lr = 0.06

# Parameters initialization
weights = np.random.normal(0, 0.1, 11)
biais = random.normalvariate(0, 0.1)

m = X_train_for_log_regr.shape[0]
for epoch in range(1000):

    # Forward pass
    Z = np.dot(X_train_for_log_regr, weights) + biais
    A = 1 / (1 + np.exp(-Z))

    # Loss Computation
    J = np.sum(-(y_train * np.log(A) + (1 - y_train) * np.log(1 - A))) / m

    # Gradient computation
    dZ = A - y_train
    dw = np.dot(dZ, X_train_for_log_regr) / m
    db = np.sum(dZ) / m

    # Update weights
    weights = weights - lr * dw
    biais = biais - lr * db

    if epoch % 10 == 0:
        print("epoch %s - loss %s" % (epoch, J))

# making Prediction using logistic regression
preds = []
probs = []
for feats in X_test_for_log_regr:
    z = np.dot(feats, weights) + biais
    a = 1 / (1 + np.exp(-z))
    probs.append(a)
    if a > 0.5:
        preds.append(1)
    elif a <= 0.5:
        preds.append(0)

"""(c) K-fold cross validation for logistic regression"""


def log_regr_pred(X_test_for_log_regr):
    preds_for_log_regr = []
    probs_for_log_regr = []
    for feats in X_test_for_log_regr:
        z = np.dot(feats, weights) + biais
        a = 1 / (1 + np.exp(-z))
        probs_for_log_regr.append(a)
        if a > 0.5:
            preds_for_log_regr.append(1)
        elif a <= 0.5:
            preds_for_log_regr.append(0)
    return preds_for_log_regr


def evaluate_algorithm_for_logistic_regression(dataset, algorithm, n_folds):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    myindex = 0
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy.pop()
        test_set_standard = sc.transform(test_set)
        predicted = algorithm(test_set_standard)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        print('logistic regression fold is %d and accuracy: %.3f' % (myindex, accuracy))
        myindex = myindex + 1
        scores.append(accuracy)
    return scores


n_folds = 10
scores_lr = evaluate_algorithm_for_logistic_regression(
    winedataset, log_regr_pred, n_folds)
print('Scores for logistic regression: %s' % scores_lr)
print('Mean Accuracy for logistic regression: %.3f%%' %
      (sum(scores_lr)/float(len(scores_lr))))

"""(d) Accuracy, Confusion matrix, ROC curve and F1 Score for logistic regression"""

conf_mat_log_reg = confusion_matrix(y_test, preds)
print('Logistic Regression Confussion Matrix')
print(conf_mat_log_reg)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_log_reg, fmt='.0f', annot=True)

# define metrics
fpr_lr, tpr_lr, _ = metrics.roc_curve(y_test,  probs)

# create ROC curve
plt.plot(fpr_lr, tpr_lr, marker='.', label='ROC curve for Logistic Regression')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()

# plot both ROC curves in one plot
plt.plot(fpr_nb, tpr_nb, label='ROC curve for Naive Bayes')
plt.plot(fpr_lr, tpr_lr, label='ROC curve for Logistic Regression')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
# f1 score using logistic regression
f1_log_regr = f1_score(y_test, preds)
print('F1 score using logistic regression: %f' % f1_log_regr)

# logistic regression accuracy score
logistic_regression_accuracy_score = metrics.accuracy_score(y_test, preds)
logistic_regression_accuracy_score

# Final Result of both naive Bayes and logistic regression for better comparision
print("**********************************************************")
print("**********************************************************")
print("*****Final Result to find the best Model******")
print("Comparision between Navie Bayes and logistic regression results")

print('Scores for different folds Naive Bayes : %s' % scores_nb)
print('Scores for logistic regression in different folds: %s' % scores_lr)

print('Mean Accuracy for Naive Bayes : %.3f%%' %
      (sum(scores_nb)/float(len(scores_nb))))
print('Mean Accuracy for logistic regression: %.3f%%' %
      (sum(scores_lr)/float(len(scores_lr))))
print('F1 score using Naive Bayes: %f' % f1ScoreForNB)
print('F1 score using logistic regression: %f' % f1_log_regr)
