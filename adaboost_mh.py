import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from utils import test_model, print_results, write_results_to_file

def split_multi_label_data(X, y):
    X2, y2, extra_col = [], [], []
    for i in range(y.shape[0]):
        for j in range(y[i].size):
            X2.append(X[i])
            extra_col.append(j)
            y2.append(y[i, j] if y[i, j] == 1 else 0)

    X2 = np.array(X2)
    extra_col = np.array(extra_col).reshape(-1, 1)
    X2 = np.concatenate((X2, extra_col), axis=1)
    y2 = np.array(y2)

    return X2, y2


data_train = loadarff('Yelp/yelpTrain.arff')
data_train = pd.DataFrame(data_train[0])
data_train = data_train.astype(int)

data_test = loadarff('Yelp/yelpTest.arff')
data_test = pd.DataFrame(data_test[0])
data_test = data_test.astype(int)

cols = data_train.columns
y_cols = []
for c in cols:
    if str(c).__contains__('Is') and not str(c).__contains__('IsRating'):
        y_cols.append(c)

cols.drop(y_cols)
X_train, y_train = data_train[cols].values, data_train[y_cols].values
X_test, y_test = data_test[cols].values, data_test[y_cols].values

X_train, y_train = split_multi_label_data(X_train, y_train)
X_test, y_test = split_multi_label_data(X_test, y_test)



accs_train, f1s_train = [], []
accs_val, f1s_val = [], []

for depth in range(1, 31):
    tree = DecisionTreeClassifier(max_depth=depth)
    acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(tree, X_train, y_train, X_test, y_test)
    print("Decesion Tree with depth = ", depth)
    #print_results(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
    mean_acc_train = np.mean(acc_train)
    mean_f1_train = np.mean(f1_train)
    mean_acc_val = np.mean(acc_val)
    mean_f1_val = np.mean(f1_val)
    accs_train.append(mean_acc_train)
    f1s_train.append(mean_f1_train)
    accs_val.append(mean_acc_val)
    f1s_val.append(mean_f1_val)


plt.plot(range(1, 31), accs_train, label='mAcc train')
plt.plot(range(1, 31), f1s_train, label='mF1-score train')
plt.plot(range(1, 31), accs_val, label='mAcc val')
plt.plot(range(1, 31), f1s_val, label='mF1-score val')
plt.xticks(range(1, 31), range(1, 31))
plt.xlabel('Depth')
plt.grid()
plt.legend()
plt.show()


# choosing depth with the highest validation f1-score
best_depth = 1
max_f1_val = 0
for depth in range(1, 31):
    if f1s_val[depth] > max_f1_val:
        max_f1_val = f1s_val[depth]
        best_depth = depth

# choose manually
best_depth = 5


# SVM
from sklearn.svm import SVC
svm = SVC()
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(svm, X_train, y_train, X_test, y_test)
print("SVM")
print_results(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMH.txt')



# Decision tree
tree = DecisionTreeClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(tree, X_train, y_train, X_test, y_test)
print("Decesion Tree")
print_results(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMH.txt')



# Boosting
adab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=best_depth), algorithm='SAMME')
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(adab, X_train, y_train, X_test, y_test)
print("Boosting")
print_results(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMH.txt')



# Random forest
forest = RandomForestClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(forest, X_train, y_train, X_test, y_test)
print("Random Forest")
print_results(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMH.txt')

