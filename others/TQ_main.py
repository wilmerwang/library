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

print(pretrainedmodels.model_names)

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, rootpath, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for tmpfolder in os.listdir(rootpath):
            basepath = rootpath + '/' + tmpfolder + '/'
            if 'normal' in tmpfolder:
                label = 0
            elif 'tumor' in tmpfolder:
                label = 1

            for tmpfile in os.listdir(basepath):
                imgs.append((basepath + tmpfile, label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)

class MyDataset2(Dataset):
    def __init__(self, rootpath, lists, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        with open(list_file) as f:
            lines = f.readlines()

        for tmp_line in lines:
            pair = tmp_line.split(' ')#filename, label, weight
            imgs.append((basepath + pair[0], int(pair[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)


data_tf = transforms.Compose([transforms.Resize(448), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#train, valid, test
train_data = MyDataset(rootpath='../Cervical-cancer-data/2019exp/patch/train/', transform=data_tf)
train_loader = DataLoader(dataset=train_data, batch_size=28, shuffle=True)

test_data = MyDataset(rootpath='../Cervical-cancer-data/2019exp/patch/valid/', transform=data_tf)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)
print("number of train: %d, number of val: %d" % (len(train_data), len(test_data)))


model_name = 'seresnet50_448'
#model_name = 'inception_v4_384'
#model_name = 'densenet161_224'
#model_name = 'resnet18_448'
if 0:
    #net = pretrainedmodels.__dict__['se_resnet50'](num_classes=2, pretrained=None)
    net = pretrainedmodels.__dict__['resnet18'](num_classes=2, pretrained=None)
    net.last_linear = nn.Linear(in_features=512, out_features=2, bias=True)
else:
    if 'seresnet50' in model_name:
        net = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=None)
        print(net.layer0.conv1.weight[:10, 0, 0, 0])
        net.load_state_dict(torch.load('se_resnet50-ce0d4300.pth'))
        print(net.layer0.conv1.weight[:10, 0, 0, 0])
        net.avg_pool = nn.AdaptiveAvgPool2d(1)
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)
        print(net.layer0.conv1.weight[:10, 0, 0, 0])
    elif  'inception_v4' in model_name:
        net = pretrainedmodels.__dict__['inceptionv4'](num_classes=1001, pretrained=None)
        print(net.last_linear.bias[:2])
        net.load_state_dict(torch.load('inceptionv4-8e4777a0.pth.1'))
        print(net.last_linear.bias[:2])
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)
        print(net.last_linear.bias[:2])
    elif  'densenet161' in model_name:
        net = pretrainedmodels.__dict__['densenet161'](num_classes=1000, pretrained='densenet161-347e6b360.pth')
        print(net.last_linear.bias[:2])
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)
        print(net.last_linear.bias[:2])
    elif  'resnet18' in model_name:
        net = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='resnet18-5c106cde.pth')
        print(net.last_linear.bias[:2])
        net.last_linear = nn.Linear(in_features=net.last_linear.in_features, out_features=2, bias=True)
        print(net.last_linear.bias[:2])

base_path = 'my_models/%s' % model_name
if not os.path.exists(base_path):
    os.mkdir(base_path)


log_file_train = 'my_models/%s_train.log' % model_name
log_file_val = 'my_models/%s_val.log' % model_name
with open(log_file_train, 'w') as f:
    pass
with open(log_file_val, 'w') as f:
    pass

net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), 1e-3)

nums_epoch = 60

losses =[]
acces = []
eval_losses = []
eval_acces = []

lr = 1e-3
for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0

    if (epoch + 1) % 20 == 0:
        lr = 1e-3
    lr *= 0.88

    optimizer = optim.SGD(net.parameters(), lr)
    print('epoch', 'lr', epoch, lr)
    net.train()

    cnt = 0
    torch.save(net.state_dict(), '%s/%d' % (base_path, epoch))
    for img , label in train_loader:
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        out = net(img)
        loss = criterion(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        cnt += 1
        if cnt%10 == 0:
            with open(log_file_train, 'a') as f:
                f.write('[epoch|iter][%d|%d],  loss: %.2f, acc: %.2f\n' % (epoch, cnt, train_loss, acc))
    eval_correct, eval_cnt = 0, 0
    net.eval()
    with torch.no_grad():
        for img , label in test_loader:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            out = net(img)
            _ , pred = out.max(1)
            num_correct = (pred==label).sum().item()
            eval_correct += num_correct
            eval_cnt += img.shape[0]
          #  print(eval_cnt)

    with open(log_file_val, 'a') as f:
        f.write('epoch[%d],  val_acc: %.2f\n' % (epoch, eval_correct/eval_cnt))


