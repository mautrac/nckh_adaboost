import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from utils import test_model, print_results, write_results_to_file

#matplotlib.use("Qt5Agg")

plt.ion()

df = pd.read_csv('bank/bank-full.csv', delimiter=';')
df.head()
cols = df.columns

data = df.copy()

label_encoder = LabelEncoder()
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Correlation checking

# pearson correlation does not work with categorical data
# plt.figure(figsize=(20, 20))
# corr_matrix = data.corr().round(3)
# sns.heatmap(corr_matrix, annot=True)
# plt.savefig('corr.jpg')

#chi2 test
chi2 = []
for col in categorical_cols:
    if col == 'y':
        continue
    chi2.append(chi2_contingency(pd.crosstab(data[col], data['y']))[1])


#data
X = data.values[:, :-1]
Y = data.values[:, -1]

#visualize label
g = sns.histplot(Y, discrete=True)
g.set_xticks(list(range(len(np.unique(Y)))), np.unique(Y))
g.set_xlabel('Label')


# class_weight
compute_class_weight(class_weight="balanced", classes=np.unique(Y), y=Y)
weights = compute_sample_weight(class_weight="balanced", y=Y)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=420, shuffle=True)
w_train = compute_sample_weight(class_weight='balanced', y=y_train)

kfold = StratifiedKFold(n_splits=5, random_state=420, shuffle=True)


# Logistic regression
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(max_iter=1000)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(logistic, X_train, y_train, X_test, y_test)
print("Logistic regression")
print_results(logistic, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(logistic, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboost.txt')


# SVM
from sklearn.svm import SVC
svm = SVC()
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(svm, X_train, y_train, X_test, y_test)
print("SVM")
print_results(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboost.txt')


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
best_depth = 7

# Decision tree
tree = DecisionTreeClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(tree, X_train, y_train, X_test, y_test)
print("Decision tree")
print_results(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboost.txt')


# Random forest
forest = RandomForestClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(forest, X_train, y_train, X_test, y_test)
print("Random forest")
print_results(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboost.txt')



# AdaBoost
adab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=best_depth), algorithm='SAMME')
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(adab, X_train, y_train, X_test, y_test)
print("AdaBoost")
print_results(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboost.txt')










