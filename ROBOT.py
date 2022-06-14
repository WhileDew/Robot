import numpy as np
import pandas as pd
from torch import nn

train = r"train.csv"
test = r"test.csv"
def data_precess(filename):
    df = pd.read_csv(filename)
    data = []
    for i in df:
        data.append(df[i].astype(np.float32))
    if 'CLASS' in df:
        label = np.array(df['CLASS'].astype(np.compat.long))
        
        data = np.array(data)[1:-1].T  #去掉前面序号和后面标签
    else:
        label = np.array([])
        data = np.array(data)[1:].T  #去掉前面的序号
    return data,label
train_data,train_labels = data_precess(train)
test_data,_ = data_precess(test)
#计算
train_data = (train_data+1)/2
test_data = (test_data+1)/2

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                             bootstrap=True, oob_score=False, n_jobs=None,
                             random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
scores = cross_val_score(clf, train_data, train_labels, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 4))


clf.fit(train_data,train_labels)
out = clf.predict(test_data)
sbmit = pd.read_csv(r"submission.csv")
sbmit['CLASS'] = out
sbmit.to_csv('submission.csv')


class shixu_Model1(nn.Module):
    def __init__(self):
        # super(final_Model1, self).__init__()
        self.bn0 = nn.BatchNorm1d(240)
        self.fc1 = nn.Linear(240, 64)
        self.dr1 = nn.Dropout(0.5)
        self.rl1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.dr2 = nn.Dropout(0.8)
        self.rl2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 2)
        self.dr3 = nn.Dropout(0.8)

    def forward(self, x):
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.dr1(x)

        x = self.rl1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.dr2(x)

        x = self.rl2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.dr3(x)
        return x
