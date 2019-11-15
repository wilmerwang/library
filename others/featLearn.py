import random
import csv
import numpy as np
from sklearn import linear_model
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import mnist
from torch import  nn
from torch.autograd import Variable
from torch import  optim

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1_1 = nn.Conv2d(2, 16, (4, 1))
        self.conv1_2 = nn.Conv2d(16, 16, (1, 1))
        self.conv1_3 = nn.Conv2d(16, 32, (1, 5))
        self.conv1_4 = nn.Conv2d(32, 32, (1, 1))

        self.conv2_1 = nn.Conv2d(2, 16, (1, 5))
        self.conv2_2 = nn.Conv2d(16, 16, (1, 1))
        self.conv2_3 = nn.Conv2d(16, 32, (4, 1))
        self.conv2_4 = nn.Conv2d(32, 32, (1, 1))

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x1 = F.relu(self.conv1_1(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = F.relu(self.conv1_2(x1))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.max_pool(x1)

        if 0:
            x1 = F.relu(self.conv1_3(x1))
            x1 = F.dropout(x1, p=0.5, training=self.training)
            x1 = F.relu(self.conv1_4(x1))
        x1 = x1.view(x1.shape[0], -1)

        x2 = F.relu(self.conv2_1(x))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = F.relu(self.conv2_2(x2))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.max_pool(x2)
        if 0:
            x2 = F.relu(self.conv2_3(x2))
            x2 = F.dropout(x2, p=0.5, training=self.training)
            x2 = F.relu(self.conv2_4(x2))
        x2 = x2.view(x2.shape[0], -1)

        #print(x1.shape, x2.shape)
        x_out = torch.cat((x1, x2), 1)
        x_out = F.dropout(x_out, p=0.5, training=self.training)
        x_out = self.fc(x_out)
        return x_out

def tensor2mat(X):
    N, M, P = X.shape
    out_X = X[:, :, 0]
    for k in range(P-1):
        out_X = np.hstack((out_X, X[:, :, k]))
    return out_X

def getGt(filename):
    with open(filename) as f:
        lines = f.readlines()

    N = len(lines)
    Y_gt = np.zeros(N)
    for k, tmp_line in enumerate(lines):
        Y_gt[k] = int(float(tmp_line.split(' ')[1]))
    return Y_gt

def getFeat(filename, N):
    X_feat = np.zeros((N, 2, len(filename)))
    X_probs = np.zeros((N, len(filename)))
    for cnt, tmp_file in enumerate(filename):
        with open(tmp_file) as f:
            reader = csv.reader(f)
            rows = [s for s in reader]
        for k, tmp_row in enumerate(rows):
            X_feat[k, :, cnt] = [float(s) for s in tmp_row[1:]]
        X_probs[:, cnt] = np.exp(X_feat[:, 1, cnt]) / (np.exp(X_feat[:, 0, cnt]) + np.exp(X_feat[:, 1, cnt]))
    return X_feat, X_probs

def estVals(X_probs, Y_gt):
    if len(X_probs.shape) == 1:
        print(np.mean((X_probs > 0.5) == Y_gt))
        return np.mean((X_probs > 0.5) == Y_gt)
    else:
        ret = []
        for k in range(X_probs.shape[1]):
            ret.append(np.mean((X_probs[:, k] > 0.5) == Y_gt))
            print(np.mean((X_probs[:, k] > 0.5) == Y_gt))
        return ret

criterion = nn.CrossEntropyLoss()


model_names = ['my_models_resnet18_448_14', 'my_models_densenet161_224_13', 'my_models_inception_v4_448_11', 'my_models_seresnet50_448_14', 'my_models_resnet18_996_14']
#model_names = ['my_models_resnet18_448', 'my_models_inception_v4_336']
data_names = ['val', 'test']

filenames = {tmp_key: [] for tmp_key in data_names}
for tmp_data in data_names:
    for tmp_model in model_names:
        for i in range(2):
            for j in range(2):
                if i == 2:
                    filenames[tmp_data].append('val_ret2/%s_5_8_%d%d_%s.csv' % (tmp_model, 0, j, tmp_data))
                else:
                    filenames[tmp_data].append('val_ret2/%s_%d%d_%s.csv' % (tmp_model, i, j, tmp_data))
#model_names = [model_names[1]]

Y_gt = {tmp_key: None for tmp_key in data_names}
X_feat, X_probs =  {tmp_key: None for tmp_key in data_names}, {tmp_key: None for tmp_key in data_names}


ret, ret2 ={}, {}
for tmp_data in data_names:
    print(tmp_data)
    Y_gt[tmp_data] = getGt('dataset3/%s_lists.txt' % tmp_data)
    X_feat[tmp_data], X_probs[tmp_data] = getFeat(filenames[tmp_data], Y_gt[tmp_data].shape[0])
    ret[tmp_data] = estVals(X_probs[tmp_data], Y_gt[tmp_data])
    ret2[tmp_data] = estVals(np.mean(X_probs[tmp_data], 1), Y_gt[tmp_data])
1/0
clf = xgb.XGBClassifier(max_depth=7,  min_child_weight=4)
#clf = linear_model.LogisticRegression(C=1e2)
#clf = RandomForestClassifier(max_depth=7, n_estimators = 133)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(8), random_state=1)

if 0:
    X_val = X_probs['val']
    X_test = X_probs['test']
else:
    X_feat_tensor, X_probs_tensor = {}, {}
    if 0:
        X_val = tensor2mat(X_feat['val'])
        X_test = tensor2mat(X_feat['test'])
    else:
        for tmp_key in data_names:
            X_feat_tensor[tmp_key] = np.zeros((X_feat[tmp_key].shape[0], 2, 4, len(model_names)))
            X_probs_tensor[tmp_key] = np.zeros((X_feat[tmp_key].shape[0], 4, len(model_names)))
            for k in range(len(model_names)):
                X_feat_tensor[tmp_key][:, :, :, k] = X_feat[tmp_key][:, :, k*4:(k+1)*4]
                X_probs_tensor[tmp_key][:, :, k] = X_probs[tmp_key][:, k*4:(k+1)*4]

        if 0:
            X_val = np.vstack((X_feat_tensor['val'][:, :, 0, :], \
                               X_feat_tensor['val'][:, :, 1, :], \
                               X_feat_tensor['val'][:, :, 2, :], \
                               X_feat_tensor['val'][:, :, 3, :]))
            X_test = np.vstack((X_feat_tensor['test'][:, :, 0, :], \
                               X_feat_tensor['test'][:, :, 1, :], \
                               X_feat_tensor['test'][:, :, 2, :], \
                               X_feat_tensor['test'][:, :, 3, :]))

            X_val = tensor2mat(X_val)
            X_test = tensor2mat(X_test)
            for tmp_key in data_names:
                Y_gt[tmp_key] = np.hstack((Y_gt[tmp_key], Y_gt[tmp_key], Y_gt[tmp_key], Y_gt[tmp_key]))

        if 0:
            if 1:
                X_val = tensor2mat(np.median(X_feat_tensor['val'], 2))
                X_test = tensor2mat(np.median(X_feat_tensor['test'], 2))
            else:
                X_val = np.median(X_probs_tensor['val'], 1)
                X_test = np.median(X_probs_tensor['test'], 1)


cmp_results = np.zeros((10, 50))
X_feat_all = np.vstack((X_feat_tensor['val'], X_feat_tensor['test']))
Y_gt_all = np.hstack((Y_gt['val'], Y_gt['test']))

X_val = tensor2mat(X_feat['val'])
X_test = tensor2mat(X_feat['test'])
X_all = np.vstack((X_val, X_test))

for ptr_cv in range(10):
    idx_val, idx_test = [], []
    for k in range(Y_gt_all.shape[0]):
        if (ptr_cv + k) % 10==0:
            idx_test.append(k)
        else:
            idx_val.append(k)

    X_val = X_all[idx_val, :]
    X_test = X_all[idx_test, :]
    Y_gt = {'val': Y_gt_all[idx_val], 'test': Y_gt_all[idx_test]}
    X_feat_tensor = {'val': X_feat_all[idx_val, :, :, :], 'test': X_feat_all[idx_test, :, :, :]}
    print(Y_gt['val'].shape, Y_gt['test'].shape)

    for kkk, ensemble_name in enumerate(['xgb', 'rf']):
         if ensemble_name == 'xgb':
             clf = xgb.XGBClassifier(max_depth=5,  min_child_weight=4)
     #clf = linear_model.LogisticRegression(C=1e2)
         elif ensemble_name == 'rf':
             clf = RandomForestClassifier(max_depth=5, n_estimators = 100)
     #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(8), random_state=1)

         clf.fit(X_val, Y_gt['val'])
         Y_val_pred = clf.predict_proba(X_val)
         Y_test_pred = clf.predict_proba(X_test)

         print('ensemble val %s' % ensemble_name)
         estVals(Y_val_pred[:, 1], Y_gt['val'])
         print('ensemble test %s' % ensemble_name)
         cmp_results[ptr_cv, kkk] = estVals(Y_test_pred[:, 1], Y_gt['test'])


    net = Model().cuda().train()
    lr = 2e-4
    #optimizer = optim.SGD(net.parameters(), lr)
    optimizer = optim.Adam(net.parameters(), lr)
    for iter in range(30000):
        idx = random.sample(range(X_feat_tensor['val'].shape[0]), 64)
        idx_aug = random.sample(range(4), 4)
       # idx_model = random.sample(range(5), 5)
        #out = net(torch.FloatTensor(X_feat_tensor['val'][idx, :, :, :][:, :, idx_aug, :][:, :, :, idx_model]).cuda())
        out = net(torch.FloatTensor(X_feat_tensor['val'][idx, :, :, :][:, :, idx_aug, :]).cuda())
        label = torch.LongTensor(Y_gt['val'][idx]).cuda()
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % 500 == 0:
            acc = 1 - torch.mean(torch.abs((out[:, 1] > out[:, 0]).float() - label.float()))
            print(iter, lr, loss, acc)
            net.eval()
            with torch.no_grad():
                out_test = net(torch.FloatTensor(X_feat_tensor['test'][:, :, :, :]).cuda()).cpu()
                label_test = torch.FloatTensor(Y_gt['test'])
                acc = 1 - torch.mean(torch.abs((out_test[:, 1] > out_test[:, 0]).float() - label_test))
                print(acc)
            net.train()

            if iter%5000 == 0:
                cmp_results[ptr_cv, int(2+iter/5000)] = acc
        if (iter+1) % 5000 == 0:
            pass
            #lr *= 0.9
            #optimizer = optim.SGD(net.parameters(), lr)
    if 0:
        net.eval()
        with torch.no_grad():
            out_test = net(torch.FloatTensor(X_feat_tensor['test'][:, :, :, :]).cuda()).cpu()
            label_test = torch.FloatTensor(Y_gt['test'])
            acc = 1 - torch.mean(torch.abs((out_test[:, 1] > out_test[:, 0]).float() - label_test))
        cmp_results[ptr_cv, 2] = float(acc)

np.savetxt('ensemble_ret.txt', cmp_results)
1/0
clf.fit(X_val, Y_gt['val'])
Y_val_pred = clf.predict_proba(X_val)
Y_test_pred = clf.predict_proba(X_test)

print('ensemble val')
estVals(Y_val_pred[:, 1], Y_gt['val'])
print('ensemble test')
estVals(Y_test_pred[:, 1], Y_gt['test'])
estVals(Y_test_pred[:, 1].reshape(4, -1).transpose(), Y_gt['test'][:5342])
#X_cat_probs = tensor2mat(X_probs['val'][:, None, :])
1/0
keys = ['test']

X_all, Y_all, X_probs_all = {}, {}, {}

for tmp_key in keys:
    with open('list_%s.txt' % tmp_key) as f:
        lines = f.readlines()

    N = len(lines)
    X_all[tmp_key], Y_all[tmp_key] = np.zeros((N, 8, len(model_names))), np.zeros(N)
    X_probs_all[tmp_key] = np.zeros((N, 4, len(model_names)))
    for k, tmp_line in enumerate(lines):
        Y_all[tmp_key][k] = int(tmp_line.split(' ')[-1])

    for cnt, tmp_name in enumerate(model_names):
        with open('val_ret2/%s_%s.csv' % (tmp_name, tmp_key)) as f:
            reader = csv.reader(f)
            rows = [s for s in reader]
        for k, tmp_row in enumerate(rows):
            X_all[tmp_key][k, :, cnt] = [float(s) for s in tmp_row[1:]]

    for cnt in range(len(model_names)):
        for k in range(int(X_all[tmp_key].shape[1]/2)):
            X_probs_all[tmp_key][:, k, cnt] = np.exp(-X_all[tmp_key][:, 2*k, cnt]) / (np.exp(-X_all[tmp_key][:, 2*k, cnt]) + np.exp(-X_all[tmp_key][:, 2*k+1, cnt]))

clf = xgb.XGBClassifier(max_depth=2, min_child_weight=3)

X_train = tensor2mat(X_all['valid'])
X_test = tensor2mat(X_all['test'])
clf.fit(X_train, Y_all['valid'])
Y_test_pred = clf.predict_proba(X_test)
np.mean((Y_test_pred[:, 0]>0.5) == Y_all['test'] )
1/0

clf.fit(np.hstack((X_train[:, :, 0], X_train[:, :, 1])), Y_train)
X_train, X_test = X_all[::2, :, :], X_all[1::2, :, :]
Y_train, Y_test = Y_all[::2], Y_all[1::2]


clf.fit(np.hstack((X_train[:, :, 0], X_train[:, :, 1])), Y_train)
Y_test_pred = clf.predict_proba(np.hstack((X_test[:, :, 0], X_test[:, :, 1])))
Y_train_pred = clf.predict_proba(np.hstack((X_train[:, :, 0], X_train[:, :, 1])))
print(np.mean((Y_test_pred[:, 1]>0.5) == Y_test))
print(np.mean((Y_train_pred[:, 1]>0.5) == Y_train))
1/0
1/0
