import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

data = pd.read_csv('Advertising.csv')
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)

r = Ridge()
ridge = r.fit(x_train, y_train)
print("Training_set score:{}".format(ridge.score(x, y)))
print("ridge.coef_:{}".format(ridge.coef_))
print("ridge.intercept_: {}".format(ridge.intercept_))
order = y_test.argsort(axis=0)
y_test = y_test.values[order]
x_test = x_test.values[order, :]
y_predict = r.predict(x_test)
mse = np.average((y_predict - np.array(y_test)) ** 2)
rmse = np.sqrt(mse)
print('MSE =', mse)
print('RMSE =', rmse)
plt.figure(facecolor='w')
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label=u'real data')
plt.plot(t, y_predict, 'b-', linewidth=2, label=u'predicted data')
plt.legend(loc='upper right')
plt.title(u'predict sales by ridge regression', fontsize=18)
plt.grid(b=True)
plt.show()

x_T = np.transpose(x_train)
num = 20000
k_mat = np.linspace(-10000, num - 1 - 10000, num=num)
beta = np.zeros([num, 3])
for k in range(num):
    I = np.eye(x_train.shape[1])
    x_inversel = (np.dot(x_T, x_train) + k_mat[k] * I)
    x_inverse = np.linalg.inv(x_inversel)
    x2 = np.dot(x_inversel, x_inverse)
    beta[k, :] = np.dot(x_inverse, np.dot(x_T, y_train))
print(beta)
plt.plot(beta)
plt.show()

a = [0.05263455, 0.25295774, 0.00258252]
y = np.dot(x_test, a)
plt.plot(y_test, 'r-', linewidth=2, label=u'real data')
plt.plot(y_predict, 'b-', linewidth=2, label=u'predicted data')
plt.plot(y)
plt.show()
