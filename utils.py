import matplotlib.pyplot as plt
from collections import defaultdict
import os
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from tqdm import tqdm
import sys
import warnings
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import time

warnings.filterwarnings('ignore')

# early_stopping
class EarlyStopping:
    def __init__(self, patience=10**3, verbose=False, path="check_oco.pt"):

        """引数:最小値の非更新数カウンタ、表示設定、モデル格納path"""
        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_min = -(np.Inf)   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_min_acc, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        # score = val_loss
        score = val_min_acc

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_min_acc, model)  #記録後にモデルを保存してスコア表示する
        elif score <= self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_min_acc, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_min_acc, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'training ft increased ({self.val_min:.6f} --> {val_min_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_min = val_min_acc  #その時のlossを記録する


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


def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True



def randomInCircle(centerX,centerY,dim, num_sample):
    datax,datay = [],[]
    for i in range(num_sample):
        u,v = np.random.uniform(0,1,(dim,1))
        theta = 2 * math.pi * u
        rad = math.sqrt(v)
        x = rad * math.cos(theta) + centerX
        y = rad * math.sin(theta) + centerY
        datax.append(x)
        datay.append(y)
    return datax,datay

 ## 45度回転した正方形内に一様
def randomInRotatedSquare(centerX, centerY, dim, num_sample):
    datax, datay = [], []
    theta = -math.pi/4  # 45度回転
    for i in range(num_sample):
        x = np.random.uniform(centerX - dim/2, centerX + dim/2)
        y = np.random.uniform(centerY - dim/2, centerY + dim/2)
        # 45度回転する
        x_rot = (x-centerX) * math.cos(theta) - (y-centerY) * math.sin(theta) + centerX
        y_rot = (x-centerX) * math.sin(theta) + (y-centerY) * math.cos(theta) + centerY
        datax.append(x_rot)
        datay.append(y_rot)
    return datax, datay


from sklearn.neural_network import MLPClassifier

'''オンライン予測のフィードバックの定義'''
class Feedbuck():
    # def __init__(self):
    #     keshi = [0.5,0.5,0.5]
    def ACC(TP,num):
        return TP/num

    def Binary_ACC(acc,theta):
        if (acc) >= theta:
            return 1
        else:
            return 0
    
def Network_init(model,path,device):
    # network = torch.load(path)
    # optimizer = optim.SGD(network.parameters(), lr=0.001,momentum=0.9)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model


'''学習開始(初期化作業)'''
# def OP_init(model,train_loader,weight_tmp,optimizer,criterion,device):
#     for _, data in enumerate((train_loader)):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels= data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         outputs = model(inputs) #model = NeuralNetwork()
#         loss = criterion(outputs, labels)
#         loss = torch.dot(loss,weight_tmp)
#         loss.backward()
#         optimizer.step()
#     print("OP_init Finish")

def OP_init(model,train_loader,weight_tmp,num_samples,optimizer,criterion,device):
    weight_tmp = weight_tmp.to('cpu').detach().numpy().copy()
    r = np.array([-1]*(num_samples))
    for _, data in enumerate(train_loader):
        inputs, labels, idx = data
        # labels = labels.reshape(-1)
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
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


def train(model,train_loader,num_samples,weight_tmp,optimizer,criterion,max_epoch,gamma,log,device):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    r = np.array([0]*(num_samples))
    model.train()
    weight_tmp = weight_tmp.to('cpu').detach().numpy().copy()
    end = time.time()

    for epoch in range(max_epoch):
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        for images, labels,idx  in tqdm(train_loader, leave=False):
            images, labels= images.to(device), labels.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs,dim = 1)
            # pred = torch.max(outputs, 1)[1].tolist()
            rt = (pred == labels)
            rt = rt.to('cpu').detach().numpy().copy().tolist()
            r[idx] = rt
            inst_weight = (torch.from_numpy(weight_tmp[idx].astype(np.float32)).clone()).to(device)
            loss = criterion(outputs, labels)
            loss = torch.dot(loss,inst_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)

        print(f"-----------------total_epoch:{epoch}------------------")
        print("train_loss:{}".format(train_loss))

        ft = np.array(weight_tmp) @ np.array(r) 

        # DNN のearly_stoppingの条件
        if ft >=  (1/2) + gamma:
            # f.append(np.dot(weight_l,rt))
            print("Satisfied with W.L Definition : {}".format(epoch))
            acc1, _ = accuracy(outputs, labels, topk=(1, 1))
            top1.update(acc1[0], images.size(0))
            batch_time.update(time.time() - end)
            output = (
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss:.4f} ({loss:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                batch_time=batch_time,loss=train_loss, top1=top1))
            print(output)
            log.write(output + '\n')
            end = time.time()
            break
        elif epoch == (max_epoch-1):
            print("Couldn't Satisfied with W.L Definition")
            sys.exit()

    return acc1,model,r,epoch

def class_wise_acc(model,loader,device):
    y_preds = []
    true_label = []
    model.eval()

    with torch.no_grad():
        for _, data in enumerate((loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels,idx= data
            inputs = inputs.to(device)
            
            predicted = model(inputs) 

            predicted = torch.max(predicted, 1)[1]
            y_preds.extend(predicted.cpu().numpy())
            true_label.extend(labels.cpu().numpy())
    
        cf = confusion_matrix(true_label,y_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)

        cls_acc = cls_hit / cls_cnt
    model.train()
    # return class_acc_list[0]
    return cls_acc,y_preds

'''Hedge Algorithm'''
def Hedge(weight_tmp,rt,N,T,device):
    eta = ((8*(math.log(N)))/(T))**(1/2)
    down = 0
    for i in range(N):
        down +=  (weight_tmp[i].item()*math.exp(-(eta*rt[i])))

    for i,item in enumerate(rt):
        weight = weight_tmp[i].item()
        
        #Hedge Algorithmの実装
        weight_tmp[i] = ((weight*math.exp(-(eta*item)))/down)

    #損失関数の重みの更新(weight_tmp)
    weight_tmp = torch.tensor(weight_tmp)
    weight_tmp = weight_tmp.to(device)
    return weight_tmp

def worst_val_idx(acc_list):
    acc_list = np.array(acc_list)
    idx = acc_list.argmin(axis=1)
    val = acc_list.min(axis=1)
    
    # n = val.argmax()
    n = max([i for i,x in enumerate(val) if x == max(val)])
    
    worst = val[n]
    return worst,n,idx

#二次元配列（リスト）を転置
def transposition(matrix):
    matrix = np.array(matrix).T
    matrix = matrix.tolist()
    return matrix

def make_label(loader):
    label = []
    for i in loader:
        _,labels = i
        #tensor to list
        label.append(labels.tolist())
    return label

def min_max(x, axis=None):
    x1 = (np.array(x).T)[0]
    x2 = (np.array(x).T)[1]
    min1 = x1.min(axis=axis, keepdims=True)
    max1 = x1.max(axis=axis, keepdims=True)
    result1 = (x1-min1)/(max1-min1)

    min2 = x2.min(axis=axis, keepdims=True)
    max2 = x2.max(axis=axis, keepdims=True)
    result2 = (x2-min2)/(max2-min2)
    result = []
    for i in range(len(x1)):
        result.append([result1[i],result2[i]])

    return result

def instance_wise_acc(model,loader,device):
    y_preds,true_label,r = [],[],[]
    model.eval()
    with torch.no_grad():
        for inputs, labels, idx  in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            true_label.extend(labels.cpu().numpy())
            predicted = model(inputs)
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

def true_false_r(y_pred,labels):
    r = []
    #T epoch終了した弱学習器の各instnceの正解数を求める
    for item in zip(labels,y_pred):
        if item[0] == item[1]:
            r.append(1)
        else:
            r.append(0)
    return r

def voting(ht):
    h_var = []
    # ht = treatment(ht)
    for m in range(len(ht)):
        count = Counter((ht[m]))
        majority = count.most_common()
        h_var.append(majority[0][0])
    h_var = transposition(h_var)
    return h_var

def voting_w_conf(ht):
    h_var = []  # 多数決の結果
    h_conf = [] # 出現回数リスト
    for row in ht:
        count = Counter(row)
        h_var.append(max(count, key=count.get))  # 最頻値を取得
        classes = sorted(count.keys())  # クラスを自動取得（多クラス対応）
        h_conf.append([count.get(cls, 0) for cls in classes])  # 出現回数
    
    return h_var, h_conf


# Input : Weight 1*x 
def weight_show(weight,path):

    weight_s = transposition(weight)

    plt.rcParams['font.size'] = 14
    # plt.rcParams['font.family'] = 'Times New Roman'
    
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
    # plt.title("Online Prediction")
    # plt.title("Ours")
    # ax1.legend()
    colors  = ['r','g','b','y']

    #データ点のプロット結果
    for i in range(len(weight[0])):
        print(i)
        plt.plot(np.arange(len(weight_s[0])),  weight_s[i], color = colors[i],label = "class : {}".format(i+1))

    ax1.legend(loc = "lower right", fontsize = 10)
    plt.savefig(path,dpi=400)
    return True

## Adding
def make_label_data_clsnum(loader):
    label = []
    cls_num = []
    datas = []
    # make_label
    for i in loader:
        data,labels = i
        #tensor to list
        label.extend(labels.tolist())
        datas.extend(data.tolist())
    
    n = len(set(label))
    # make_cls_num
    for i in range(n):
        cls_num.append(label.count(i))
        
    return label,datas,cls_num

# cls_num_list = [100,100,100,10]
def data_plot(loader):
    _,data,cls_num = make_label_data_clsnum(loader)
    sns.set()
    # sns.set_style(style='whitegrid')
    colors=['r','b','g','y']
    #データ点のプロット結果
    tmp_idx = 0
    for i in range(len(cls_num)):
        if i == 0:
            tmp_idx = 0
        else:
            tmp_idx = sum(cls_num[0:i])
        data_plot = np.array(data[tmp_idx:tmp_idx+cls_num[i]])
        x = data_plot.transpose()[0]
        y = data_plot.transpose()[1]
        plt.scatter(x, y, color=colors[i],s=10,label = "class : {}".format(i+1))
    plt.legend()
    plt.show()

    return True

## show confusion matrix
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None):
    fig = plt.subplots()
    cm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,square=True, cbar=True, annot=True, cmap='Reds',fmt='.5g')
    plt.yticks(rotation=0)
    plt.title = title
    plt.xlabel("Pre", fontsize=13, rotation=0)
    plt.ylabel("GT", fontsize=13)
    plt.show()
    # fig.tight_layout()
    return True


def calc_acc(label,pred):
    # print(pred)
    class_acc_list = []
    cf = confusion_matrix(label,pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    ## Adding
    cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
    cls_acc = np.around(cls_acc ,decimals=4)
    # cls_acc = cls_hit / cls_cnt

    class_acc_list.append(cls_acc.tolist())
    return class_acc_list[0]


def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def area_show_w_sample_size(h,coord,label,sample_weight,path):
    
    sample_weight = [i*100 for i in sample_weight]
    X = np.arange(0,1.0,(0.01/3))
    Y = np.arange(0,1.0,(0.01/3))

    x_0,y_0 = [],[]
    x_1,y_1 = [],[]

    color = ["b","r"]
    colors = []

    for i in range(len(label)):
        colors.extend(color[label[i]]) 

    cnt = 0
    for i in range(len(X)):
            for j in range(len(Y)):
                if h[cnt] == 0:
                    x_0.append(X[i])
                    y_0.append(Y[j])
                elif h[cnt] == 1:
                    x_1.append(X[i])
                    y_1.append(Y[j])
                else :
                    print('stop')
                    sys.exit(1)
                cnt += 1

    plt.rcParams['font.size'] = 14

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig = plt.figure()
    ax1 = plt.subplot(111)
    
    # グラフの上下左右に目盛線を付ける。
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')

    #多数決の分類結果（背景）
    im = plt.scatter(x_0,y_0,color = color[0],s = 0.1,alpha = 0.3)
    im = plt.scatter(x_1,y_1,color = color[1],s = 0.1,alpha = 0.3)

    # データ点のプロット結果
    im = plt.scatter(coord.T[0],coord.T[1], c=colors, s=sample_weight,marker = "o")

    #label付けを行うための表示
    im = plt.scatter([], [], label='class:1', color = color[0]) 
    im = plt.scatter([], [], label='class:2', color = color[1]) 

    plt.savefig(path, dpi = 400)
    return im