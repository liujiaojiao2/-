#预测波士顿房价,加载数据集，训练模型，验证模型，预测结果
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
#1\读取数据集
train_data=pd.read_csv("HousingData/train.csv")
#2\数据预处理
X=train_data.drop(['MEDV'],axis=1) #X为特征值
Y=train_data['MEDV'] #Y为标签值
#填补缺失值，处理NaN数值
simple=SimpleImputer(strategy='mean')
X=simple.fit_transform(X)
#对特征值进行标准化
scale=StandardScaler()
x_scale=scale.fit_transform(X)
#3、划分训练集和测试集
x_train,x_val,y_train,y_val=train_test_split(x_scale,Y,test_size=0.2,random_state=54)
#4、训练模型
#线性回归模型-----------------------
model_LR=LinearRegression()
model_LR.fit(x_train,y_train)
#SGD回归器-------------------------
model_SGD=SGDRegressor(max_iter=1000,tol=1e-3,learning_rate='invscaling',eta0=0.01,warm_start=True)#随机梯度下降回归器
train_losss=[]
test_losses=[]
for epoch in range(4):
    model_SGD.partial_fit(x_train,y_train) #使用训练集数据更新模型参数
    #计算训练集上的损失函数
    y_pred=model_SGD.predict(x_train)
    loss=mean_squared_error(y_train,y_pred)
    train_losss.append(loss)
    #计算验证集上的损失函数
    y_pred=model_SGD.predict(x_val)
    loss=mean_squared_error(y_val,y_pred)
    test_losses.append(loss)

#绘制损失函数
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(range(len(train_losss)),train_losss,label='training loss_SGD')
plt.plot(range(len(test_losses)),test_losses,label='validation loss_SGD')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss_SGD')
plt.legend()
plt.grid(True)
plt.show()
#5、验证模型
y_predict_LR=model_LR.predict(x_val)
msq_LR=mean_squared_error(y_val,y_predict_LR) #计算均方误差
print(f"使用LinearRegression模型均方误差为：{msq_LR}")
#--------------------
y_predict_SGD=model_SGD.predict(x_val)
msq_SGD=mean_squared_error(y_val,y_predict_SGD)
print(f"使用SGDRegressor模型均方误差为：{msq_SGD}")
#6、测试集
    #处理测试数据
test_data=pd.read_csv("HousingData/test.csv")
test_data=simple.transform(test_data)
test_data=scale.transform(test_data)
y_predict_LR=model_LR.predict(test_data)
y_predict_SGD=model_SGD.predict(test_data)
#7\预测结果
#submission=pd.DataFrame({'MEDV':y_predict_LR})
submission=pd.DataFrame({'MEDV':y_predict_SGD})
submission.to_csv("evaluate_kit/submission.csv")