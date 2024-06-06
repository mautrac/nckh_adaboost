import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


def write_results_to_file(model, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val, file):
    with open(file, 'a') as f:
        f.write(str(model))
        f.write("\n")
        f.write("acc_train: " + str(np.mean(acc_train)))
        f.write("\n")
        f.write("f1_train: " + str(np.mean(f1_train)))
        f.write("\n")
        f.write("acc_test: " + str(np.mean(acc_test)))
        f.write("\n")
        f.write("f1_test: " + str(np.mean(f1_test)))
        f.write("\n")
        f.write("acc_val: " + str(np.mean(acc_val)))
        f.write("\n")
        f.write("f1_val: " + str(np.mean(f1_val)))
        f.write("\n")
        f.write("\n")


def print_results(model, acc_train, f1_train, acc_test, f1_test, acc_val, f1_val):
    print(model)
    print("acc_train: ", np.mean(acc_train))
    print("f1_train: ", np.mean(f1_train))
    print("acc_test: ", np.mean(acc_test))
    print("f1_test: ", np.mean(f1_test))
    print("acc_val: ", np.mean(acc_val))
    print("f1_val: ", np.mean(f1_val))


def test_model(modeler, X_train, y_train, X_test, y_test):
    acc_train = []
    f1_train = []
    acc_test = []
    f1_test = []
    acc_val = []
    f1_val = []
    #kfold = KFold(n_splits=5, random_state=420, shuffle=True)
    classes = np.unique(y_train).size
    metric = 'binary' if classes == 2 else 'micro'

    kfold = StratifiedKFold(n_splits=5, random_state=420, shuffle=True)
    for (train_val_idx, test_val_idx) in kfold.split(X_train, y_train):

        model = modeler.fit(X_train[train_val_idx], y_train[train_val_idx])

        y_hat = model.predict(X_train[train_val_idx])
        acc_train.append(accuracy_score(y_train[train_val_idx], y_hat))
        f1_train.append(f1_score(y_train[train_val_idx], y_hat, average=metric))

        y_hat = model.predict(X_test)
        acc_test.append(accuracy_score(y_test, y_hat))
        f1_test.append(f1_score(y_test, y_hat, average=metric))

        y_hat = model.predict(X_train[test_val_idx])
        acc_val.append(accuracy_score(y_train[test_val_idx], y_hat))
        f1_val.append(f1_score(y_train[test_val_idx], y_hat, average=metric))

    return acc_train, f1_train, acc_test, f1_test, acc_val, f1_val