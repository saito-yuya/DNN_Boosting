import argparse
from datasets import CustomDataset
import models
from utils import * 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))



def get_args():
    parser = argparse.ArgumentParser(description='hyper parameter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', default="0", help='device')
    parser.add_argument('--dataset_type', default="moon", help='dataset_type')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mlp')
    parser.add_argument('--gamma', type = float ,default=0.10000,help='gamma')
    parser.add_argument('--eps', type = float, default=5e-4,help='small margin of gamma')
    parser.add_argument('--min_size', type = int, default=50,help='min_size')
    parser.add_argument('--patience', type = int, default=10**3,help='patience')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes ')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--max_epoch', default=10**4, type=int, metavar='N',
                    help='number of maximum epochs to run')
    parser.add_argument('--seed', type = int, default=0,help='seed')
    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
    parser.add_argument('--train_rule', default="None", type=str, help='train rule')
    parser.add_argument('--root_log',type=str, default='log')
    parser.add_argument('--root_model', type=str, default='checkpoint')
    parser.add_argument('--store_name', type=str, default='None')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    torch_fix_seed(args.seed)
    gamma =  args.gamma
    min_size = args.min_size
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.store_name = '_'.join([args.dataset_type,args.loss_type,str(args.seed)])
    prepare_folders(args)

    print("gamma:",gamma)
    print("min_size:",min_size)
    print("seed:",args.seed)
    print("device:",args.gpu)
    log_dict = {"train_acc_round":[],"round_train_time":[],"num_epochs":[],"sample_weight":[]}
    


    if args.dataset_type == 'moon':
        X, y = make_moons(n_samples=args.min_size*args.num_classes, shuffle=True, noise=0.2, random_state=args.seed)
        
    elif args.dataset_type == 'circle':
        X, y = make_circles(n_samples=args.min_size*args.num_classes, noise=0.2, factor=0.5, random_state=args.seed)
    else:
        print('=== Choose dataset type ===')
        sys.exit(1)
    
    X  = min_max(X)

    data_training = CustomDataset(X, y)

    train_loader = DataLoader(data_training, batch_size=len(X),shuffle=False) ## バッチサイズ＝データ数
    check_loader = DataLoader(data_training, batch_size=len(X),shuffle=False)
    np.save(f'{args.root_log}/train_X_{args.dataset_type}.npy',X)
    np.save(f'{args.root_log}/train_y_{args.dataset_type}.npy',y)

    # plot
    colors = {0:"#FF0000", 1:"#0000FF"}

    for color_index, (point_x, point_y) in zip(y, X):
        plt.plot(point_x, point_y, color=colors[color_index], linestyle='dashed', marker='o', markersize=10)
    plt.savefig(f'traindata_plot_{args.dataset_type}.png', dpi=400)
    # plt.show()
    plt.close()
    # ===

    # init log for training
    log_training_txt = open(os.path.join(args.root_log, args.store_name, 'log_train.txt'), 'w')
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing_txt = open(os.path.join(args.root_log, args.store_name, 'log_test.txt'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))


    model_path = os.path.join(args.root_model, args.store_name)


    print("=> creating model '{}'".format(args.arch))
    if  args.arch == 'resnet18':
        model =  models.resnet18(in_channels = args.num_in_channels, num_classes=args.num_classes)

    elif args.arch == 'resnet50':
        model =  models.resnet50(in_channels = args.num_in_channels, num_classes=args.num_classes)
    
    elif args.arch == 'mlp':
        model = models.MLPNet(args.num_classes).to(device)

    else:
        raise NotImplementedError
    torch.save(model.state_dict(), model_path +f'/{args.loss_type}_model.pt')

    earlystopping = EarlyStopping(patience=args.patience, verbose=True, path = model_path +  f'/{args.loss_type}_model.pt')

    num_sample = len(X)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    round_num = math.ceil(2*math.log(num_sample)/(gamma)**2) 
    print(f"round_num :{round_num}")
    # round_num = 3 ## for debug

    weight_tmp = torch.tensor([1/num_sample]*num_sample).to(device)
    log_dict["sample_weight"].append(weight_tmp.to('cpu').detach().numpy().copy().tolist())


    criterion = nn.CrossEntropyLoss(reduction= "none").to(device)

    OP_init(model,train_loader,weight_tmp,num_sample,optimizer,criterion,device)

    for t in tqdm(range(round_num)):
    ## Train 
        model = Network_init(model,model_path +  f'/{args.loss_type}_model.pt',device)
        # start = time.time()
        train_acc,weak_model,r,epoch = train(model,train_loader,num_sample,weight_tmp,optimizer,criterion,args.max_epoch,gamma,log_training,device)
        # end = time.time()

        torch.save(model.state_dict(), model_path +f'/weak-l({t}).pt')
        weight_tmp = Hedge(weight_tmp,r,num_sample,round_num,device)

        
        log_dict["sample_weight"].append(weight_tmp.to('cpu').detach().numpy().copy().tolist())
        log_dict["train_acc_round"].append(train_acc.to('cpu').detach().numpy().copy().tolist()[0])
        # log_dict["round_train_time"].append(round(end - start,5))
        log_dict["num_epochs"].append(epoch)

    np.save(os.path.join(args.root_log, args.store_name, 'sample_weight.npy'), log_dict["sample_weight"])
    np.save(os.path.join(args.root_log, args.store_name, 'train_acc_round.npy'), log_dict["train_acc_round"])
    # np.save(os.path.join(args.root_log, args.store_name, 'round_train_time.npy'), log_dict["round_train_time"])
    np.save(os.path.join(args.root_log, args.store_name, 'num_epochs_each_round.npy'), log_dict["num_epochs"])

    weak_preds = []

    ## Train_acc
    for t in tqdm(range(round_num)):
        model.load_state_dict(torch.load(model_path +f'/weak-l({t}).pt'))
        model  = model.to(device)
        _,tmp_h = class_wise_acc(model,check_loader,device)
        weak_preds.append(tmp_h)

    ht = transposition(weak_preds)
    h = voting(ht)

    train_labels = data_training.__getlabels__()
    train_acc = calc_acc(train_labels,h)

    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    out_cls_acc = 'Class Train Accuracy: %s'%(np.array2string(np.array(train_acc), separator=',', formatter={'float_kind':lambda x: "%.4f" % x}))
    log_training.write(out_cls_acc + '\n')
    log_training.flush()

    print("--------------------------------------Finish---------------------------------------------------")









