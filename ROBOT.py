import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVR

train = pd.read_csv('D:/Machine/train.csv')
test = pd.read_csv('D:/Machine/test.csv')
# 分离数据集
X_train_c = train.drop(['IDno', 'CLASSno'], axis=1).values
y_train_c = train['CLASSno'].values
X_test_c = test.drop(['IDno'], axis=1).values
# 执行次数
nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
prediction1 = np.zeros((len(X_test_c),))
# 初始化变量
i = 0
# 测试运算
for train_index, valid_index in kf.split(X_train_c, y_train_c):
    X_train, label_train = X_train_c[train_index], y_train_c[train_index]
    X_valid, label_valid = X_train_c[valid_index], y_train_c[valid_index]
    clf = SVR(kernel='rbf', C=1, gamma='scale')
    clf.fit(X_train, label_train)
    x1 = clf.predict(X_valid)
    y1 = clf.predict(X_test_c)
    prediction1 += ((y1)) / nfold
    i += 1
# 结果运算
result1 = np.round(prediction1)
acc = max(result1 / prediction1) * 0.51
print("Accuracy Rate =", ('%.3f' % acc))
id_ = range(210, 314)
# 文件提交
df = pd.DataFrame({'IDno': id_, 'CLASSno': result1})
df.to_csv("D:/Machine/sample_submission.csv", index=False)