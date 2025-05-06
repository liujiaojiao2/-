"""地址相关性判断"""
#1、加载json数据
import json
from difflib import SequenceMatcher

import editdistance
import matplotlib.pyplot as plt
import numpy as np
from Levenshtein import distance

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.font_manager import FontProperties
from scipy.sparse import hstack
from sympy.physics.units import grams
from torch.backends.opt_einsum import strategy
from torch.distributed.pipelining import pipeline

train_data=[]
with open('AddressRelation/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        train_data.append(json.loads(line))
train_data=pd.DataFrame(train_data)

#2、数据预处理，清洗数据,异常值，空白值
    #去除首尾空格
train_data['query']=train_data['query'].str.strip()
for i in range(len(train_data)):
    for j in range(len(train_data['candidate'][i])):
        train_data['candidate'][i][j]['text']=train_data['candidate'][i][j]['text'].strip()
#3\特征提取
# def calculate_distance(query,candidate):
#     return distance(query,candidate)
# train_data['distance']=train_data.apply(lambda row: [calculate_distance(row['query'],cand['text']) for cand in row['candidate']],axis=1)
#加入模糊匹配
def get_fuzzy_matching_features(query,candidate):
    # 模糊匹配 ratio (difflib)
    ratio = SequenceMatcher(None, query, candidate).ratio()
    return [ratio]
all_text=[]
labels=[]
distance_features=[]
fuzzy_features=[]
#计算编辑距离
def calculate_distance(query,candidate):
    # return 1/(distance(query,candidate)+1)
    edit_distance=editdistance.eval(query,candidate)
    norm_edit=edit_distance/max(len(query),len(candidate))
    return norm_edit
for _,row in train_data.iterrows():
    for cand in row['candidate']:
        all_text.append(row['query'] + ' ' +cand['text'])
        labels.append(cand['label'])
        distance_features.append(calculate_distance(row['query'],cand['text']))
        fuzzy_features.append(get_fuzzy_matching_features(row['query'],cand['text']))

#文本向量化
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=30000)
X=vectorizer.fit_transform(all_text)
#由于训练数据类别不平衡
# 不匹配     46798    部分匹配    25527      完全匹配     4813
# ，需要重采样，增加部分匹配，减少不匹配
#多采样部分匹配

#融合编辑距离和向量化后的文本
X=hstack([X,np.array(distance_features).reshape(-1,1)])
#融合模糊匹配
X=hstack([X,np.array(fuzzy_features)])
# #label转化为one-hot编码
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# labels=le.fit_transform(labels)
    #划分训练集、验证集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=0.2,random_state=42)
#使用smote过采样
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
smote=SMOTE(sampling_strategy={'部分匹配':30000,'完全匹配':10000})
under=RandomUnderSampler(sampling_strategy={'不匹配':40000})
pipeline=Pipeline([('smote',smote),('under',under)])
X_train,y_train=pipeline.fit_resample(X_train,y_train)
#输出标签类别分布结果
label_count_train=pd.Series(y_train).value_counts()
label_count_test=pd.Series(y_test).value_counts()
print(f"训练集类别分布为：{label_count_train}")
print(f"测试集类别分布为：{label_count_test}")

#4\模型训练,评估模型
#选择分类模型
#朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()


#xgboost模型
from xgboost import XGBClassifier
#model=XGBClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#模型评估，使用F1-score评估
from sklearn.metrics import f1_score, classification_report

print(classification_report(y_test,y_pred,target_names=["完全匹配", "部分匹配", "不匹配"]))
#f1=f1_score(y_test,y_pred,average='macro')
#设置字体为中文支持的字体
font=FontProperties(fname='/System/Library/Fonts/Supplemental/Songti.ttc')
plt.rcParams['font.family']=font.get_name()

#计算混淆矩阵
import seaborn as sns
from sklearn.metrics import confusion_matrix
labels=['完全匹配', '部分匹配', '不匹配']
cm=confusion_matrix(y_test,y_pred,labels=labels)
    #可视化混淆矩阵
    #设置图形大小
plt.figure(figsize=(10,6))
    #使用heatmap绘制混淆矩阵
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels)
    #添加标题和标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.show()
f1=f1_score(y_test,y_pred,average='weighted')
print('F1-score:',f1)
#测试集测试数据
#逐行加载json格式的测试集
test_data=[]
with open('AddressRelation/test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))
test_data=pd.DataFrame(test_data)
#处理文件格式
test_data_text=[]
test_fearture=[]
test_fuzzyy_features=[]
for _,row in test_data.iterrows():
    for cand in row['candidate']:
        test_data_text.append(row['query'] + ' ' +cand['text'])
        test_fearture.append(calculate_distance(row['query'],cand['text']))
        test_fuzzyy_features.append(get_fuzzy_matching_features(row['query'],cand['text']))
test_data_tfidf=vectorizer.transform(test_data_text)
test_fearture=np.array(test_fearture).reshape(-1,1)
test_fuzzyy_features=np.array(test_fuzzyy_features)
test_input=hstack([test_data_tfidf,test_fearture,test_fuzzyy_features])
test_pred=model.predict(test_input)

#5\保存结果到submission文件
submission_data=[]
intdex=0
for _,row in test_data.iterrows():
    result={
        "text_id":row['text_id'],
        "query":row['query'],
        "candidate":[]
    }
    for i in range(len(row['candidate'])):
        result_cand={
            "text":row['candidate'][i]['text'],
            "label":test_pred[intdex]
        }
        intdex+=1
        result['candidate'].append(result_cand)
    submission_data.append(result)
#写入submission文件
with open('evaluate_kit/submission.jsonl', 'w', encoding='utf-8') as f:
    for result in submission_data:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')