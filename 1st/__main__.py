import pandas as pd
from io import StringIO
from sklearn import linear_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    csv_data = 'square_feet,price\n150,6450\n200,7450\n250,8450\n300,9450\n350,11450\n400,15450\n600,18450\n'
    # 读入dataframe
    df = pd.read_csv(StringIO(csv_data))
    print(df)
    x = df['square_feet'].reshape(-1, 1)
    y = df['square_feet']
    # 建立线性回归模型
    regr = linear_model.LinearRegression()
    # 拟合
    regr.fit(x, y)  # 注意此处.reshape(-1, 1)，因为X是一维的！
    # 不难得到直线的斜率、截距
    a, b = regr.coef_, regr.intercept_
    # 给出待预测面积
    area = 238.5
    # 方式1：根据直线方程计算的价格
    print("price=", a * area + b)
    # 方式2：根据predict方法预测的价格
    print("price predicted=", regr.predict(area))
    # 画图
    # 1.真实的点
    plt.scatter(x, y, color='blue', label='real price')
    # 2.拟合的直线
    plt.plot(x, regr.predict(x), color='red', linewidth=4, label='predicted price')
    plt.xlabel('area')
    plt.ylabel('price')
    plt.legend(loc='lower right')
    plt.show()
