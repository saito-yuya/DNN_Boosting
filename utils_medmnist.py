from collections import defaultdict
import os
import time
from os import TMP_MAX
import torch
from torch import nn
# from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from itertools import chain 
import sklearn.metrics as metrics
from tqdm import tqdm
import sys
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import warnings
import torchvision.datasets as datasets

warnings.filterwarnings('ignore')

from sklearn.utils.multiclass import unique_labels

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
    
class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def make_label_clsnum(loader):
    label = []
    cls_num = []
    for i in loader:
        _,labels,_ = i
        labels = labels.reshape(-1)

        label.extend(labels.tolist())
    n = len(set(label))
    for i in range(n):
        cls_num.append(label.count(i))
        
    return label,cls_num

def torch_fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def Network_init(model,path,device):
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model

def OP_init(model,train_loader,weight_tmp,num_samples,optimizer,criterion,device):
    weight_tmp = weight_tmp.to('cpu').detach().numpy().copy()
    r = np.array([-1]*(num_samples))
    for _, data in enumerate(train_loader):
        inputs, labels, idx = data
        labels = labels.reshape(-1)
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        pred = torch.argmax(outputs,dim = 1)
        rt = (pred == labels)
        rt = rt.to('cpu').detach().numpy().copy().tolist()
        r[idx] = rt
        inst_weight = (torch.from_numpy(weight_tmp[idx].astype(np.float32)).clone()).to(device)
        loss = criterion(outputs, labels)
        loss = torch.dot(loss,inst_weight) 
        # loss = torch.dot(loss,batch_weight) 
        loss.backward()
        optimizer.step()
    print("OP_init Finish")
    return True


def class_wise_acc(model,loader,device):
    class_acc_list,y_preds,true_label = [],[],[]
    model = model.to(device) 
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate((loader)):
                inputs, labels,_ = data
                labels = labels.reshape(-1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                predicted,_ = model(inputs) 
                predicted = torch.max(predicted, 1)[1]
                y_preds.extend(predicted.cpu().numpy())
                true_label.extend(labels.cpu().numpy())
        cf = confusion_matrix(true_label,y_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
        # cls_acc = cls_hit / cls_cnt
        class_acc_list.append(cls_acc)
    model.train()
    return class_acc_list[0],y_preds,true_label,cls_cnt

def calc_acc(label,pred):
    class_acc_list = []
    cf = confusion_matrix(label,pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
    cls_acc = np.around(cls_acc ,decimals=4)
    # cls_acc = cls_hit / cls_cnt
    class_acc_list.append(cls_acc.tolist())
    return class_acc_list

def class_wise_acc_h(y_pred,labels):
    ans = defaultdict(int)
    for item in zip(labels,y_pred):
        if item[0] == item[1]:
            ans[item[0]] += 1
    return ans 

def instance_wise_acc(model,loader,device):
    y_preds,true_label,r = [],[],[]
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels,_ = inputs.to(device), labels.to(device)
            labels = labels.reshape(-1)

            true_label.extend(labels.cpu().numpy())
            predicted,_ = model(inputs)
            predicted = torch.max(predicted, 1)[1]
            y_preds.extend(predicted.cpu().numpy())
            c = (predicted == labels).squeeze()

            for i in range(inputs.shape[0]):
                if c[i]:
                    r.append(1)
                else:
                    r.append(0)
    model.train()
    return r,y_preds,true_label

def train(model,train_loader,num_samples,weight_tmp,optimizer,criterion,max_epoch,gamma,log,device):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    r = np.array([-1]*(num_samples))
    model.train()
    weight_tmp = weight_tmp.to('cpu').detach().numpy().copy()
    end = time.time()

    for epoch in range(max_epoch):
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        for images, labels,idx  in tqdm(train_loader, leave=False):
            labels = labels.reshape(-1)

            # batch_weight = weight_tmp[count*(images.shape[0]):(count+1)*(images.shape[0])]
            # batch_weight = batch_weight.to(device)
            # labels = labels.reshape(-1)
            images, labels= images.to(device), labels.to(device)
            outputs,_ = model(images)
            pred = torch.argmax(outputs,dim = 1)
            rt = (pred == labels)
            rt = rt.to('cpu').detach().numpy().copy().tolist()
            r[idx] = rt
            inst_weight = (torch.from_numpy(weight_tmp[idx].astype(np.float32)).clone()).to(device)
            loss = criterion(outputs, labels)
            loss = torch.dot(loss,inst_weight) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)
        # rt,y_preds,_ = instance_wise_acc(model,train_loader,device)

        print("-----------------total_epoch:{}------------------".format(epoch))
        print("train_loss:{}".format(train_loss))

        weight_l = weight_tmp.tolist()

        ft = 0
        for n in range(num_samples):
            ft +=  weight_l[n]*r[n]

        # DNN のearly_stoppingの条件
        if ft >=  (1/2) + gamma:
            # f.append(np.dot(weight_l,rt))
            print("Satisfied with W.L Definition : {}".format(epoch))
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 2))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            output = (
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss:.4f} ({loss:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_time=batch_time,loss=train_loss, top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            end = time.time()
            break
        elif epoch == (max_epoch-1):
            print("Couldn't Satisfied with W.L Definition")
            sys.exit()

    return acc1,model,r

def Hedge(weight_tmp,rt,classes,round_num,device):
    eta = ((8*(math.log(classes)))/(round_num))**(1/2)
    down = 0
    for i in range(classes):
        down +=  (weight_tmp[i].item()*math.exp(-(eta*rt[i])))
    for i,item in enumerate(rt):
        weight = weight_tmp[i].item()
        weight_tmp[i] = ((weight*math.exp(-(eta*item)))/down)

    weight_tmp = torch.tensor(weight_tmp)
    weight_tmp = weight_tmp.to(device)
    return weight_tmp

def transposition(matrix):
    matrix = np.array(matrix).T
    matrix = matrix.tolist()
    return matrix

#input:各モデル（弱学習器）の予測ラベル
#output:多数決後（強学習器）の予測ラベル
def voting(ht):
    h_var = []
    # ht = treatment(ht)
    for m in range(len(ht)):
        count = Counter((ht[m]))
        majority = count.most_common()
        h_var.append(majority[0][0])
    h_var = transposition(h_var)
    return h_var

# votingの入力が違うversion
def ensemble(ht,keep):
    keep.append(ht)
    #  ht = treatment(ht)
    keep = transposition(keep)
    h_var = []
    for m in range(len(keep)):
        count = Counter((keep[m]))
        majority = count.most_common()
        h_var.append(majority[0][0])
    h_var = transposition(h_var)
    return h_var

# inp : model_i　における予測list,正解ラベル
# out : model_iまでの予測多数決のaccuracy list
def best_N(out_list,y_true):
    keep = []
    acc_list = []
    for i in range(len(out_list)):
        out = out_list[i]
        if i != 0:
            res = ensemble(out,keep)
        else:
            res = out
        acc = calc_acc(y_true,res)
        keep.append(out)
        acc_list.append(acc[0])
    return acc_list

def worst_val_idx(acc_list):
    acc_list = np.array(acc_list)
    idx = acc_list.argmin(axis=1)
    val = acc_list.min(axis=1)
    # n = val.argmax()
    n = max([i for i,x in enumerate(val) if x == max(val)])
    worst = val[n]
    return worst,n,idx

# Input : Weight 1*x 
def weight_show(weight,classes,path):

    weight_s = transposition(weight)

    sns.set()
    sns.set_style(style='whitegrid')
    sns.set_palette("husl",classes)
    
    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig = plt.figure()
    ax1 = plt.subplot(111)
    
    # グラフの上下左右に目盛線を付ける。
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    
    # 軸のラベルを設定する。
    ax1.set_xlabel('Round t')
    ax1.set_ylabel('Weight') 

    #データ点のプロット結果
    for i in range(len(weight[0])):
        plt.plot(np.arange(len(weight_s[0])),  weight_s[i], 'o-', lw =1, label = "class : {}".format(i+1))

    ax1.legend(loc = "lower left", fontsize = 10)
    
    plt.savefig(path,dpi=400)
    plt.close()

    return True

def calc_acc_ave(label,pred):
    # print(pred)
    ave_list = []
    cf = confusion_matrix(label,pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    ## Adding
    # cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
    # cls_acc = np.around(cls_acc ,decimals=4)
    # cls_acc = cls_hit / cls_cnt
    ave = sum(cls_hit)/sum(cls_cnt)

    ave_list.append(ave.tolist())
    return ave_list

def ave(out_list,y_true):
    keep = []
    ave_list = []
    for i in range(len(out_list)):
        out = out_list[i]
        if i != 0:
            res = ensemble(out,keep)
            # print(res)
        else:
            res = out
        acc = calc_acc_ave(y_true,res)
        keep.append(out)
        ave_list.append(acc[0])
    return ave_list