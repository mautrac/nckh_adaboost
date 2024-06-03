import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight, compute_class_weight
from utils import test_model, print_results, write_results_to_file

df = pd.read_excel('dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx')
data = pd.read_excel('dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx')
columns = data.columns
x_cols = columns.drop('Class')

label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'].values)

# chuẩn hóa dữ liệu theo z-score
standard_scaler = StandardScaler()
data[x_cols] = standard_scaler.fit_transform(data[x_cols].values)

# ma trận hệ số tương quan
cormat = data.corr()
round(cormat)
sns.heatmap(cormat, annot=True)

X = data.values[:, :-1]
y = data.values[:, -1]

# visualize label
g = sns.histplot(y, discrete=True)
g.set_xticks(list(range(len(np.unique(y)))), np.unique(y))
g.set_xlabel('Label')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
w_train = compute_sample_weight(class_weight='balanced', y=y_train)

# compute class weight

compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
compute_sample_weight(class_weight='balanced', y=y)

kfold = StratifiedKFold(n_splits=5, random_state=420, shuffle=True)



accs_train, f1s_train = [], []
accs_val, f1s_val = [], []

for depth in range(1, 31):
    tree = DecisionTreeClassifier(max_depth=depth)
    acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(tree, X_train, y_train, X_test, y_test)
    print("Decesion Tree with depth = ", depth)
    # print_results(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
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

# SVM
svm = SVC()
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(svm, X_train, y_train, X_test, y_test)
print("SVM")
print_results(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostM1.txt')


# Decision tree
tree = DecisionTreeClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(tree, X_train, y_train, X_test, y_test)
print("Decesion Tree")
print_results(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostM1.txt')


# Boosting
adab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=best_depth), algorithm='SAMME')
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(adab, X_train, y_train, X_test, y_test)
print("Boosting")
print_results(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostM1.txt')


# Random forest
forest = RandomForestClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(forest, X_train, y_train, X_test, y_test)
print("Random Forest")
print_results(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostM1.txt')


