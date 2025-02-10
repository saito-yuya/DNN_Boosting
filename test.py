from utils import * 
from score import *
import argparse
from datasets import CustomDataset
import models 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def get_args():
    parser = argparse.ArgumentParser(description='hyper parameter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', default="0",help='device')
    parser.add_argument('--dataset_type', default="moon", help='dataset_type')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mlp')
    parser.add_argument('--eps', type = float, default=5e-4,help='small margin of gamma')
    parser.add_argument('--gamma', type = float ,default=0.1000,help='gamma')
    parser.add_argument('--min_size', type = int, default=50,help='min_size')
    parser.add_argument('--test_min_size', type = int, default=100000,help='test_min_size')
    parser.add_argument('--seed', type = int, default=0,help='seed')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes ')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--max_epoch', default=10**4, type=int, metavar='N',
                    help='number of maximum epochs to run')
    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
    parser.add_argument('--train_rule', default="None", type=str, help='train rule')
    parser.add_argument('--root_log',type=str, default='log')
    parser.add_argument('--root_model', type=str, default='checkpoint')
    parser.add_argument('--store_name', type=str, default='None')

    return parser.parse_args()


X = np.arange(0,1.0,(0.01/3))
Y = np.arange(0,1.0,(0.01/3))

'''データセットの定義'''
class AreaDataset(Dataset):
    def __init__(self):
        self.data_list = []
        for i in range(len(X)):
            for j in range(len(Y)):
                #順伝搬
                inp = np.array([X[i],Y[j]]) 
                inp = torch.from_numpy(inp.astype(np.float32)).clone()
                self.data_list.append(inp)
        # self.data_list = self.data_list[0]

    def __len__(self):
        return len(list(self.data_list))
        # return 
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data_list[idx])

def area_test(model,loader1):
    model.eval()
    tmp_h = []
    for inputs in (loader1):
        inputs = inputs.to(device)
        out = model(inputs)
        y_pred = torch.max(out, 1)[1].tolist()
        tmp_h.extend(y_pred)
    model.train()
    return tmp_h

def area_show(h,sample_weight,path):
    x_0,y_0,x_1,y_1 = [], [], [], []

    #一次元リストhを色分けして二次元にプロットする
    cnt = 0
    for i in range(len(X)):
            for j in range(len(Y)):
                if h[cnt] == 0:
                    x_0.append(X[i])
                    y_0.append(Y[j])
                elif h[cnt] == 1:
                    x_1.append(X[i])
                    y_1.append(Y[j])
                cnt += 1

    x_l = np.load(f'log/train_X_{args.dataset_type}.npy')
    y_l = np.load(f'log/train_y_{args.dataset_type}.npy')

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


    #データ点のプロット結果
    colors = {0:"#FF0000", 1:"#0000FF"}
    count = 0
    for color_index, (point_x, point_y) in zip(y_l, x_l):
        im = plt.plot(point_x, point_y, color=colors[color_index], marker='o', markersize=sample_weight[count]*500,alpha = 0.5)
        count += 1
    #多数決の分類結果（背景）
    im = plt.scatter(x_0,y_0,c="#FF0000",s = 0.1,alpha = 0.3)
    im = plt.scatter(x_1,y_1,c="#0000FF",s = 0.1,alpha = 0.3)
    plt.savefig(path + '.png',dpi = 400)

    return im

def area_show_w_conf(h,h_conf,sample_weight,path):
    x_0,y_0,x_1,y_1 = [], [], [], []
    red_c,blue_c = [],[]

    #一次元リストhを色分けして二次元にプロットする
    cnt = 0
    for i in range(len(X)):
            for j in range(len(Y)):
                if h[cnt] == 0:
                    x_0.append(X[i])
                    y_0.append(Y[j])
                    red_c.append(max(h_conf[cnt]))
                else:
                    x_1.append(X[i])
                    y_1.append(Y[j])
                    blue_c.append(max(h_conf[cnt]))
                cnt += 1

    x_l = np.load(f'log/train_X_{args.dataset_type}.npy')
    y_l = np.load(f'log/train_y_{args.dataset_type}.npy')

    plt.rcParams['font.size'] = 14
    
    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig = plt.figure()
    ax1 = plt.subplot(111)
    
    # グラフの上下左右に目盛線を付ける。
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')


    #データ点のプロット結果
    colors = {0:"#FF0000", 1:"#0000FF"}
    count = 0
    for color_index, (point_x, point_y) in zip(y_l, x_l):
        # im = plt.plot(point_x, point_y, color=colors[color_index], marker='o', markersize=sample_weight[count]*500,alpha = 0.5)
        im = plt.plot(point_x, point_y, color=colors[color_index], marker='o', markersize=5,alpha = 0.5)
        count += 1

    alpha_values_red = [(i/max(red_c))-0.5 for i in red_c]
    alpha_values_blue = [i/max(blue_c)-0.5 for i in blue_c]
    red_colors = [(1, 0, 0, alpha) for alpha in alpha_values_red]
    blue_colors = [(0, 0, 1, alpha) for alpha in alpha_values_blue]
    #多数決の分類結果（背景）
    im = plt.scatter(x_0,y_0,c=red_colors,s = 0.1)
    im = plt.scatter(x_1,y_1,c=blue_colors,s = 0.1)
    plt.savefig(path + '.png',dpi = 400)

    return im

if __name__ == '__main__':
    args = get_args()
    gamma =  args.gamma
    seed = args.seed
    min_size = args.min_size
    device = torch.device('cuda')
    args.store_name = '_'.join([args.dataset_type,args.loss_type,str(args.seed)])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    print("gamma:",gamma)
    print("seed:",seed)
    print("device:",args.gpu)

    torch_fix_seed(seed)

    print("=> creating model '{}'".format(args.arch))
    if  args.arch == 'resnet18':
        model =  models.resnet18(in_channels = args.num_in_channels, num_classes=args.num_classes)

    elif args.arch == 'resnet50':
        model =  models.resnet50(in_channels = args.num_in_channels, num_classes=args.num_classes)
    
    elif args.arch == 'mlp':
        model = models.MLPNet(args.num_classes).to(device)

    else:
        raise NotImplementedError
    model_path = os.path.join(args.root_model, args.store_name)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #バッチサイズ（今回はフルバッチ）
    batch = 5000*100
    test_set = AreaDataset()
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    # check_datasets = AreaDataset()
    # check_dataloader = DataLoader(check_datasets, batch_size=batch, shuffle=False) #generator=torch.Generator().manual_seed(42)


    #epoch Number
    T = args.max_epoch

    round_num = (sum(os.path.isfile(os.path.join(model_path, name)) for name in os.listdir(model_path)))

    train_acc_list,test_acc_list = [],[]

    ht,weak_preds = [],[]

    sample_weight = np.load(os.path.join(args.root_log, args.store_name)+'/sample_weight.npy')

    for t in tqdm(range(round_num-2)):
        sample_weight_tmp = sample_weight[t]
        model.load_state_dict(torch.load(model_path +f'/weak-l({t}).pt'))
        model  = model.to(device)
        model.eval()
        tmp_h = area_test(model,test_loader)
        ht.append(tmp_h)

    ht = transposition(ht)
    h,h_conf = voting_w_conf(ht)

    area_show_w_conf(h,h_conf,sample_weight[0], f'{args.loss_type}_decision_boundary.pdf')
    np.save(f'{args.root_log}/test_h_{args.dataset_type}.npy',h)
    np.save(f'{args.root_log}/test_confh_{args.dataset_type}.npy',h_conf)

    ## value test
    # test_min_size = args.test_min_size
    # test_min_size = 100000

    # if args.dataset_type == 'moon':
    #     test_X, test_y = make_moons(n_samples=test_min_size*2, noise=0.2, random_state=0)
    # elif args.dataset_type == 'circle':
    #     test_X, test_y = make_circles(n_samples=test_min_size*2, noise=0.2, factor=0.5, random_state=0)
    # else:
    #     print('=== Choose dataset type ===')
        
    
    # test_X  = min_max(test_X)

    # np.save(f'{args.root_log}/test_X_{args.dataset_type}.npy',test_X)
    # np.save(f'{args.root_log}/test_y_{args.dataset_type}.npy',test_y)
    ## plot
    # colors = {0:"#FF0000", 1:"#0000FF"}

    # for color_index, (point_x, point_y) in zip(test_y, test_X):
    #     plt.plot(point_x, point_y, color=colors[color_index], linestyle='dashed', marker='o', markersize=5)
    # # plt.show()
    # plt.savefig('./Toy/Ours/figure/test_data_plot.png', dpi=400)
    # plt.show()
    # plt.close()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)

    # value_dataset = CustomDataset(test_X, test_y)
    # value_loader = DataLoader(value_dataset, batch_size=batch, shuffle=False)

    # score(args,value_loader,model,round_num-1,model_path,device)

    print("--------------------------------------Finish---------------------------------------------------")






 