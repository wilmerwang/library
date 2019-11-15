import os
import numpy as np

lists = []
for tmp_file in os.listdir('.'):
    if ('prob' in tmp_file) and ('.txt' in tmp_file):
        with open(tmp_file) as f:
            lists += f.readlines()

labels = np.array([float(s.split(' ')[-1]) for s in lists])
N_normal = np.sum(labels <= 0.5)
N_tumor = np.sum(labels > 0.5)
N_tr_tumor = np.round(N_tumor * 0.8)
N_val_tumor_normal = np.round(N_tumor * 0.1)
N_test_tumor_normal = N_tumor - N_tr_tumor - N_val_tumor_normal
N_tr_normal = N_normal - N_val_tumor_normal - N_test_tumor_normal
ratio_weight = N_tr_normal / N_tr_tumor
ratio_tr_val = int(0.5 + N_tr_normal / N_val_tumor_normal)

lists_tr, lists_val, lists_test = [], [], []
cnt_normal, cnt_tumor = 0, 0
for k, tmp_line in enumerate(lists):
    pairs = tmp_line.rstrip().split()
    if float(pairs[-1]) > 0.5: #tumor
        if cnt_tumor % 10 == 0:
            lists_val.append(pairs[0] + ' 1.0 1.0\n')
        elif cnt_tumor % 10 == 1:
            lists_test.append(pairs[0] + ' 1.0 1.0\n')
        else:
            #lists_tr.append(pairs[0] + ' 1.0 %.4f\n' % ratio_weight)
            for kkk in range(int(ratio_weight+np.random.rand())):
                lists_tr.append(pairs[0] + ' 1.0 1.0\n')
        cnt_tumor += 1
    else:
        if cnt_normal % (ratio_tr_val + 2) == 0:
            lists_val.append(pairs[0] + ' 0.0 1.0\n')
        elif cnt_normal % (ratio_tr_val + 2) == 1:
            lists_test.append(pairs[0] + ' 0.0 1.0\n')
        else:
            lists_tr.append(pairs[0] + ' 0.0 1.0\n')
        cnt_normal += 1

labels_tr = np.array([float(s.split(' ')[-2]) for s in lists_tr])
labels_val = np.array([float(s.split(' ')[-2]) for s in lists_val])
labels_test = np.array([float(s.split(' ')[-2]) for s in lists_test])
print('Number train: tumor: %d, normal: %d' % (np.sum(labels_tr>0.5), np.sum(labels_tr<=0.5)))
print('Number val: tumor: %d, normal: %d' % (np.sum(labels_val>0.5), np.sum(labels_val<=0.5)))
print('Number test: tumor: %d, normal: %d' % (np.sum(labels_test>0.5), np.sum(labels_test<=0.5)))

with open('train_lists.txt', 'w') as f:
    f.writelines(lists_tr)
with open('val_lists.txt', 'w') as f:
    f.writelines(lists_val)
with open('test_lists.txt', 'w') as f:
    f.writelines(lists_test)

1/0
with open('merged_lists.txt', 'w') as f:
    f.writelines(lists)
