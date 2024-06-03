
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.arff import loadarff
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from utils import test_model, print_results, write_results_to_file

path = 'Yeast/'
data = loadarff('Yeast/yeast-train.arff')

def create():
  data = loadarff('Yeast/yeast-train.arff')
  temp = np.asarray(data[0].tolist()).astype(float)
  X_train, y_train = temp[:, :-14], temp[:, -14:]
  y_train = np.where(y_train == 0, -1, 1)

  data = loadarff('Yeast/yeast-test.arff')
  temp = np.asarray(data[0].tolist()).astype(float)
  X_test, y_test = temp[:, :-14], temp[:, -14:]
  y_test = np.where(y_test == 0, -1, 1)

  return X_train, X_test, y_train, y_test


def preprocess(class_num = 14):
  x_train, x_test, y_train, y_test = create()
  x_train_new, x_test_new, y_train_new, y_test_new = [], [], [], []
  meta_info_train = []

  train_set_len = len(x_train)
  for i in range(train_set_len):
    pos  = []
    neg  = []
    for j in range(class_num):
      x_train_new.append(np.concatenate((x_train[i], [j])))
      if y_train[i][j] == 1:
        pos.append(j)
      else:
        neg.append(j)
    meta_info_train.append((pos, neg))
  x_train_new = np.array(x_train_new)
  y_train_new = y_train.flatten()

  test_set_len = len(x_test)
  for i in range(test_set_len):
    for j in range(class_num):
      x_test_new.append(np.concatenate((x_test[i], [j])))
  x_test_new = np.array(x_test_new)
  y_test_new = y_test.flatten()

  return x_train_new, x_test_new, y_train_new, y_test_new, meta_info_train


def postprocess(x_test, y_test, clfs, alpha):
  class_num = 14
  epoch_num = len(clfs)

  predicted = [clfs[i].predict(x_test) * alpha[i] for i in range(epoch_num)]
  predicted = np.array(predicted)
  #print(predicted)
  predicted = np.sum(predicted, axis=0)
  predicted = np.array([1 if i >= 0 else -1 for i in predicted], dtype=np.int8)

  accuracy = np.sum(predicted == y_test) / len(y_test)
  f1 = f1_score(y_test, predicted, average='micro')

  print('Epoch: {}, accuracy: {}'.format(epoch_num, accuracy))
  predicted = predicted.reshape([-1, class_num]).astype(np.int8)

  return predicted, accuracy, f1


X_train, X_test, y_train, y_test, meta_info_train = preprocess(14)
class_num = 14


kfold = KFold(n_splits=5, random_state=420, shuffle=True)


def adaboost_mr(X_train, y_train, meta_info_train, epoch_num = 50, VIS = False, depth=20):

  ini_train_num = len(meta_info_train)
  alpha = np.zeros(epoch_num)
  clfs = []
  D = [] # save the weight for each sample i + label j
  pairD = {} # save the weight for each sample i + pos label j + neg label k
  sampleWeight = 1.0/float(ini_train_num)
  for i in range(ini_train_num):
    tmpD = [0]*class_num
    posWeight = sampleWeight/(2*len(meta_info_train[i][0]))
    negWeight = sampleWeight/(2*len(meta_info_train[i][1]))
    pairNum = len(meta_info_train[i][0])*len(meta_info_train[i][1])
    for j in meta_info_train[i][0]:
      tmpD[j] = posWeight
    for j in meta_info_train[i][1]:
      tmpD[j] = negWeight
    for j in meta_info_train[i][0]:
      for k in meta_info_train[i][1]:
        key = '{}_{}_{}'.format(i, j, k)
        pairD[key] = sampleWeight/pairNum
    D.extend(tmpD)

  result = []
  for i in range(epoch_num):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train, sample_weight=D)
    predicted = clf.predict(X_train)
    r = 0.0
    for s in range(ini_train_num):
      for j in meta_info_train[s][0]:
        for k in meta_info_train[s][1]:
          key = '{}_{}_{}'.format(s,j,k)
          r += pairD[key]*(predicted[s*class_num+j]-predicted[s*class_num+k])

    r = r/2.0
    a = 0.5 * np.log((1 + r) / (1 - r))
    alpha[i] = a
    clfs.append(clf)
    sum_pairD = 0
    for s in range(ini_train_num):
      for j in meta_info_train[s][0]:
        for k in meta_info_train[s][1]:
          key = '{}_{}_{}'.format(s,j,k)
          pairD[key] = pairD[key]*np.exp(0.5*a*(predicted[s*class_num+k]-predicted[s*class_num+j]))
          sum_pairD += pairD[key]

    D = []
    for s in range(ini_train_num):
      tmpD = [0]*class_num
      for j in meta_info_train[s][0]:
        for k in meta_info_train[s][1]:
          key = '{}_{}_{}'.format(s,j,k)
          pairD[key] = pairD[key]/sum_pairD
          tmpD[j] += pairD[key]/2.0
          tmpD[k] += pairD[key]/2.0
      D.extend(tmpD)

    if VIS:
      #print(alpha)
      predicted, accuracy, f1 = postprocess(X_test, y_test, clfs, alpha)
      result.append([accuracy])
      print('{} {}'.format(i + 1, accuracy))

  if VIS:
    predicted, accuracy, f1 = postprocess(X_test, y_test, clfs, alpha)
    return accuracy

  return clfs, alpha





def test_mr(X_train, y_train, X_test, y_test, meta_train_info, depth):
    acc_train = []
    f1_train = []
    acc_test = []
    f1_test = []
    acc_val = []
    f1_val = []

    for (train_val_idx, test_val_idx) in kfold.split(meta_train_info):
        temp_meta_train_info = [meta_train_info[i] for i in train_val_idx]
        temp_x_train, temp_y_train = [], []
        temp_x_val, temp_y_val = [], []
        for idx in train_val_idx:
            temp_x_train.extend(X_train[idx*class_num:(idx+1)*class_num])
            temp_y_train.extend(y_train[idx*class_num:(idx+1)*class_num])

        for idx in test_val_idx:
            temp_x_val.extend(X_train[idx*class_num:(idx+1)*class_num])
            temp_y_val.extend(y_train[idx*class_num:(idx+1)*class_num])

        clfs, alpha = adaboost_mr(temp_x_train, temp_y_train, temp_meta_train_info, epoch_num=50, VIS=False, depth=depth)

        predicted, accuracy, f1 = postprocess(X_test, y_test, clfs, alpha)
        acc_test.append(accuracy)
        f1_test.append(f1)

        predicted, accuracy, f1 = postprocess(temp_x_train, temp_y_train, clfs, alpha)
        acc_train.append(accuracy)
        f1_train.append(f1)

        predicted, accuracy, f1 = postprocess(temp_x_val, temp_y_val, clfs, alpha)
        acc_val.append(accuracy)
        f1_val.append(f1)

    return acc_train, f1_train, acc_test, f1_test, acc_val, f1_val




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
best_depth = 20



# SVM
from sklearn.svm import SVC
svm = SVC()
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(svm, X_train, y_train, X_test, y_test)
print('SVM')
print_results(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(svm, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMR.txt')


# Decision tree
tree = DecisionTreeClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(tree, X_train, y_train, X_test, y_test)
print('Decision Tree')
print_results(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(tree, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMR.txt')


# random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=best_depth)
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(forest, X_train, y_train, X_test, y_test)
print('Random Forest')
print_results(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(forest, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMR.txt')




# adaboost.MR
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_mr(X_train, y_train, X_test, y_test, meta_info_train, best_depth)
print("Multi-Label Adaboost")
print_results('Multi-Label Adaboost', acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file('Multi-Label Adaboost', acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMR.txt')

#adaboost_mr(X_train, y_train, meta_info_train, epoch_num=50, VIS=True)


# adaboost mh
from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=20), algorithm='SAMME')
acc_train, f1_train, acc_test, f1_test, acc_val, f1_val = test_model(adab, X_train, y_train, X_test, y_test)
print('AdaBoost')
print_results(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val)
write_results_to_file(adab, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, 'results_adaboostMR.txt')



