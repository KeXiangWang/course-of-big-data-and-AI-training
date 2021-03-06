import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def SVM(acc_sequence, gyr_sequence, label_sequence):
    x = np.append(gyr_sequence, acc_sequence, axis=2)
    print(x.shape)
    y = label_sequence
    x_train, x_verify, y_train, y_verify = train_test_split(x, y, train_size=0.9, test_size=0.1, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.88, test_size=0.12,
                                                        random_state=1)
    ss = StandardScaler()
    model = svm.SVC(gamma="scale")
    y_test_res = np.ones([6, len(y_test)])
    print("shape of test:", y_test_res.shape)
    for i in range(6):
        train = x_train[:, :, i]
        test = x_test[:, :, i]
        ss.fit_transform(train)
        ss.fit_transform(test)
        model.fit(train, y_train)
        y_test_res[i] = model.predict(test)
        print('accuracy_score[{}]:'.format(i), accuracy_score(y_test, y_test_res[i]))
    y_test_hat = np.ones(len(y_test))
    for i in range(len(y_test)):
        y_test_hat[i] = np.argmax(np.bincount(y_test_res[:, i].astype(np.int16)))
    print('accuracy_score:', accuracy_score(y_test, y_test_hat))
