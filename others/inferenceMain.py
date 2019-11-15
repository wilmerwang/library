import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import mnist
from torch import  nn
from torch.autograd import Variable
from torch import  optim
from torchvision import transforms
import sys
from PIL import Image
sys.path.append('pretrained-models.pytorch')
import pretrainedmodels
import csv
import copy

def loadModel(filenames):
    if 'seresnet50' in filenames[0]:
        net = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=None)
        net.avg_pool = nn.AdaptiveAvgPool2d(1)
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)
    elif 'densenet' in filenames[0]:
        net = pretrainedmodels.__dict__['densenet161'](num_classes=2, pretrained=None)
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)
    elif 'inception' in filenames[0]:
        net = pretrainedmodels.__dict__['inceptionv4'](num_classes=2, pretrained=None)
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)
    else:
        net = pretrainedmodels.__dict__['resnet18'](num_classes=2, pretrained=None)
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)

    if type([]) == type(filenames):
        state_weights = torch.load(filenames[0])
        for tmp_file in filenames[1:]:
            tmp_weights = torch.load(tmp_file)
            for tmp_key, tmp_val in tmp_weights.items():
                state_weights[tmp_key] += tmp_val

        for tmp_key, tmp_val in state_weights.items():
            state_weights[tmp_key] = state_weights[tmp_key] / len(filenames)
    else:
        state_weights = torch.load(filenames)

    net.load_state_dict(state_weights)
    net = net.cuda()
    return net

if len(sys.argv) < 2:
    model_name = 'my_models/resnet18_448'
    is_left_right, is_top_bottom = False, False
    idx = [5, 8]
    N_batch = 24
    infer_data = 'val'
else:
    model_name = 'my_models/' + sys.argv[1]
    is_left_right, is_top_bottom = int(sys.argv[2]), int(sys.argv[3])
    idx = [int(s) for s in sys.argv[4].split('_')]
    if idx[0] == -1:
        idx = idx[0]
    N_batch = int(sys.argv[5])
    infer_data = sys.argv[6]
#model_name = 'my_models/seresnet50_448'
#model_name = 'my_models/_384'

iters = os.listdir(model_name + '/')
if idx == -1:
    model_snaps = [model_name + '/' + str(np.max([int(s) for s in iters]))]
    log_names = model_snaps[0] + '_' + str(int(is_left_right)) + str(int(is_top_bottom))
else:
    model_snaps = [model_name + '/' + str(s) for s in idx]
    log_names = model_name + '/' + '_'.join([str(s) for s in idx]) + '_' + str(int(is_left_right)) + str(int(is_top_bottom))

net = loadModel(model_snaps)

img_sz = int(model_name.split('_')[-1])
data_tf = transforms.Compose([transforms.Resize(img_sz), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if 0:
    with open('list_test.txt', 'w') as f:
        for cnt, tmp_name in enumerate(['normal', 'tumor']):
            infer_data_path = '../Cervical-cancer-data/2019exp/patch/%s/%s/' % (infer_data, tmp_name)
            for k in os.listdir(infer_data_path):
                f.write('%s%s %d\n' % (infer_data_path, k, cnt))
if 0:
    with open('list_test.txt') as f:
        lines = f.readlines()[::100]
else:
    with open('dataset3/' + infer_data + '_lists.txt') as f:
        lines = f.readlines()

with open('val_ret2/%s_%s.csv' % (log_names.replace('/', '_'), infer_data), 'w') as f:
    pass

acc_one, acc_all = [], []
if N_batch == 1:
    net.eval()
else:
    net.train()

with torch.no_grad():
    with open('val_ret2/%s_%s.csv' % (log_names.replace('/', '_'), infer_data), 'a') as f:
        writer = csv.writer(f)
        ptr_batch = 0
        batch_data = torch.zeros(N_batch, 3, img_sz, img_sz)
        tmp_row, gt_label = [], []
        for cnt, tmp_line in enumerate(lines):
            strs = tmp_line.rstrip().split(' ')
            tmp_path = strs[0]
            tmp_label = strs[1]
            tmp_img = Image.open(tmp_path)
            if N_batch == 1:
                tmp_row = [tmp_path]
            else:
                tmp_row.append([tmp_path])
                gt_label.append(int(float(tmp_label)))

            if is_left_right:
                tmp_img = tmp_img.transpose(Image.FLIP_LEFT_RIGHT)
            if is_top_bottom:
                tmp_img = tmp_img.transpose(Image.FLIP_TOP_BOTTOM)

            if N_batch == 1:
                data = data_tf(tmp_img).view(1, 3, img_sz, img_sz)
                data = Variable(data).cuda()
                pred = net.forward(data)
                idx = torch.argmax(pred)
                print(cnt, tmp_label, idx, pred)
                tmp_row += [float(pred[0, 0]), float(pred[0, 1])]
                gt_label = int(float(tmp_label))
                acc_one.append(float(idx == gt_label))
                acc_all.append(float((np.sum(tmp_row[1::2]) < np.sum(tmp_row[2::2])) == gt_label))
                writer.writerow(tmp_row)
                #nn.Softmax(1)(pred)
            else:
                tmp_data = data_tf(tmp_img).view(1, 3, img_sz, img_sz)
                batch_data[ptr_batch, :, :, :] = tmp_data
                ptr_batch += 1

                if cnt >= len(lines)-1:
                    N_batch = ptr_batch

                if ptr_batch >= N_batch:
                    data = Variable(batch_data[:N_batch, :, :, :]).cuda()
                    pred = net.forward(data)
                    idx = torch.argmax(pred, 1)
                    for kkk in range(N_batch):
                        tmp_row[kkk] += [float(pred[kkk, 0]), float(pred[kkk, 1])]
                        print(cnt-N_batch+kkk+1, gt_label[kkk], idx[kkk], pred[kkk, :])

                    for kkk in range(N_batch):
                        acc_one.append(float(idx[kkk] == gt_label[kkk]))
                        acc_all.append(float((np.sum(tmp_row[kkk][1::2]) < np.sum(tmp_row[kkk][2::2])) == gt_label[kkk]))
                        writer.writerow(tmp_row[kkk])

                    ptr_batch = 0
                    batch_data = torch.zeros(N_batch, 3, img_sz, img_sz)
                    tmp_row, gt_label = [], []



print('acc_one: %.4f' % np.mean(acc_one))
print('acc_all: %.4f' % np.mean(acc_all))
