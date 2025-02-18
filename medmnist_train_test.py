
from utils_medmnist import *
import models
from opts_medmnist import parser
import medmnist
from medmnist import INFO, Evaluator
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        data, label,_ = self.dataset[index]
        return data, label, index
    def __len__(self):
        return len(self.dataset)
    @property
    def classes(self):
        return self.dataset.classes


class MLP(nn.Module):
    def __init__(self, n_hidden, n_out):
        super().__init__()
        self.l1 = nn.Linear(28*28, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_out)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.act(self.l1(x))
        h2 = self.l2(h1)
        return h2,h1
    

if __name__ == '__main__':
    args = parser.parse_args()

    gamma = args.gamma
    classes = args.classes
    torch_fix_seed(args.seed)
    args.store_name = '_'.join([args.dataset, args.arch,str(args.seed),str(args.train_sample_size),str(gamma)])
    log_dict = {"train_acc_round":[],"val_acc_ens_round":[],"train_acc_ens_round":[],"num_epochs":[],"round_train_time":[],"sample_weight":[]} #log_dict["loss_lin"].append(loss_lin.sum().item())

    prepare_folders(args)

    if args.gpu == 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(args.gpu) #使用したいGPUの番号を入れる
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("gamma:",args.gamma)
    print("device:",args.gpu)


    if args.dataset == 'mnist':
        transform = transforms.Compose([
        transforms.ToTensor(),        # テンソルに変換 & 0-255 の値を 0-1 に変換
        ]) 
    elif args.dataset == 'medmnist':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
        ])
    else :
        transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_val = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
    if args.dataset == 'cifar10':
        train_set = datasets.CIFAR10("./data",download=True,train=True, transform=transform_train)
        train_indices, val_indices = train_test_split(list(range(len(train_set.targets))), test_size=args.test_size, stratify=train_set.targets)
        train_indices = random.sample(train_indices, args.train_sample_size)
        train_indices = random.sample(train_indices, args.vals_sample_size)
        val_set = torch.utils.data.Subset(train_set, val_indices)
        val_set = DatasetWithIndex(val_set) 
        train_set = torch.utils.data.Subset(train_set, train_indices)
        train_set = DatasetWithIndex(train_set)
        test_set = datasets.CIFAR10("./data",download=True,train=False, transform=transform_val)

    elif args.dataset == 'cifar100':
        train_set = datasets.CIFAR100("./data",download=True,train=True, transform=transform_train)
        train_indices, val_indices = train_test_split(list(range(len(train_set.targets))), test_size=args.test_size, stratify=train_set.targets)
        val_set = torch.utils.data.Subset(train_set, val_indices)
        train_set = torch.utils.data.Subset(train_set, train_indices)
        test_set = datasets.CIFAR100("./data",download=True,train=False, transform=transform_val)

    elif args.dataset == 'mnist':
        train_set = datasets.MNIST("./data",download=True,train=True, transform=transform)
        train_indices, val_indices = train_test_split(list(range(len(train_set.targets))), test_size=args.test_size, stratify=train_set.targets)
        train_indices = random.sample(train_indices, args.train_sample_size)
        val_indices = random.sample(val_indices, args.val_sample_size)
        val_set = torch.utils.data.Subset(train_set, val_indices)
        val_set = DatasetWithIndex(val_set) 
        train_set = torch.utils.data.Subset(train_set, train_indices)
        train_set = DatasetWithIndex(train_set)
        test_set = datasets.MNIST("./data",download=True,train=False, transform=transform)
        test_set = DatasetWithIndex(test_set)
        
    elif args.dataset == 'medmnist':
        data_flag = args.data_flag

        info = INFO[data_flag]
        # task = info['task']
        # n_channels = info['n_channels']
        # n_classes = len(info['label'])
        MedMNIST = getattr(medmnist, info['python_class'])

        train_set = MedMNIST(split='train', transform=transform,download=True)
        train_indices = random.sample(range(len(train_set)), args.train_sample_size)
        train_set = torch.utils.data.Subset(train_set, train_indices)
        train_set = DatasetWithIndex(train_set)


        val_set = MedMNIST(split='val', transform=transform,download=True)
        val_set = DatasetWithIndex(val_set)

        # val_set = SubsetWithLabels(val_set)
        test_set = MedMNIST(split='test', transform=transform, download=True)

    else:
        warnings.warn('Dataset is not listed')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,sampler=None)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size*2,shuffle=False,num_workers=args.workers,pin_memory=True,sampler=None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*2,shuffle=False, num_workers=args.workers,pin_memory=True)

    labal,cls_num_list = make_label_clsnum(train_loader)
    num_sample = sum(cls_num_list)
    print("Train : cls_num_list",cls_num_list)

    # init log for training
    log_training_txt = open(os.path.join(args.root_log, args.store_name, 'log_train.txt'), 'w')
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing_txt = open(os.path.join(args.root_log, args.store_name, 'log_test.txt'), 'w')
    # log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    model_path = os.path.join(args.root_model, args.store_name)

    weight,weak_preds = [],[]

    print("=> creating model '{}'".format(args.arch))
    # if args.arch != 'mlp':
        # model = models.__dict__[args.arch](num_classes=classes, use_norm=False)
    if  args.arch == 'resnet18':
        model =  models.resnet18(in_channels = args.num_in_channels, num_classes=args.classes)
    elif args.arch == 'resnet50':
        model =  models.resnet50(in_channels = args.num_in_channels, num_classes=args.classes)
    elif args.arch == 'mlp':
        model = MLP(n_hidden=512, n_out=args.classes)
    else :
        raise NotImplementedError
    
    torch.save(model.state_dict(), model_path + '/check_point.pt')
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()


    #round Number(The number of weak-learner)
    round_num = math.ceil(2*math.log(num_sample)/(gamma)**2)
    # round_num = 3

    weight_tmp = torch.tensor([1/num_sample]*num_sample)
    log_dict["sample_weight"].append(weight_tmp.to('cpu').detach().numpy().copy().tolist())
    weight.append(weight_tmp.to('cpu').detach().numpy().copy().tolist())
    

    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    OP_init(model,train_loader,weight_tmp,num_sample,optimizer,criterion,device)

    for t in range(round_num):
        model = Network_init(model,model_path + '/check_point.pt',device)
        start = time.time()
        train_acc,weak_model,rt = train(model,train_loader,num_sample,weight_tmp,optimizer,criterion,args.max_epoch,gamma,log_training,device)
        end = time.time()
        
        torch.save(weak_model.state_dict(), model_path + f'/weak_model({t}).pt')

        weight_tmp = Hedge(weight_tmp,rt,num_sample,round_num,device)

        log_dict["sample_weight"].append(weight_tmp.to('cpu').detach().numpy().copy().tolist())
        log_dict["train_acc_round"].append(train_acc.to('cpu').detach().numpy().copy().tolist()[0])
        log_dict["round_train_time"].append(round(end - start,5))

    print("############################ Finish Main roop ##############################")
    np.save(os.path.join(args.root_log, args.store_name, 'sample_weight.npy'), log_dict["sample_weight"])
    np.save(os.path.join(args.root_log, args.store_name, 'train_acc_round.npy'), log_dict["train_acc_round"])
    np.save(os.path.join(args.root_log, args.store_name, 'round_train_time.npy'), log_dict["round_train_time"])

    ## validation 
    val_label,cls_num_list = make_label_clsnum(val_loader)
    # val_label = val_set.labels

    for t in tqdm(range(round_num),leave=False):
        tmp_h = []
        model.load_state_dict(torch.load(model_path + f'/weak_model({t}).pt'))
        model  = model.to(device)
        check_accuracy,tmp_h,_,_  = class_wise_acc(model,val_loader,device)
        weak_preds.append(tmp_h)

    res = best_N(weak_preds,val_label)
    val_ave_accuracy_round = ave(weak_preds,val_label)
    valid_n = np.array(val_ave_accuracy_round).argmax(axis=0)

    # log_dict["val_acc_ens_round"].append(train_acc.to('cpu').detach().numpy().copy().tolist()[0])
    np.save(os.path.join(args.root_log, args.store_name, 'val_acc_ens_round.npy'), val_ave_accuracy_round)

    print("val_accuracy :",val_ave_accuracy_round[valid_n])
    print("number of models :",valid_n)

    with open(os.path.join(args.root_log, args.store_name, 'log_train.txt'), "w") as train_f:
        print(f"dataset : {args.dataset}",file=train_f)
        print(f"gamma = {gamma}",file=train_f)
        print(f"seed = {args.seed}",file=train_f)
        print("===================Validation===================",file=train_f)
    
    with open(os.path.join(args.root_log, args.store_name, 'log_train.txt'),"a") as train_f:
        print("best number of models :",valid_n+1,file=train_f)
        print("val_best_accuracy :",res[valid_n],file=train_f)
        print("val_Average_accuracy :",val_ave_accuracy_round[valid_n],file=train_f)

    
#################
    ## Test
    test_label,cls_num_list = make_label_clsnum(test_loader)
    weak_preds = []

    for t in tqdm(range(valid_n+1)):
        tmp_h = []
        model.load_state_dict(torch.load(model_path + f'/weak_model({t}).pt'))
        check_accuracy,tmp_h,_,_  = class_wise_acc(model,test_loader,device)
        weak_preds.append(tmp_h)

    ## 全モデルに対するaccuracy_list
    res = best_N(weak_preds,test_label)
    test_ave_accuracy_round = ave(weak_preds,test_label)
    test_h = voting(np.array(weak_preds).T)

    np.save(os.path.join(args.root_log, args.store_name, 'test_acc_ens_round.npy'), test_ave_accuracy_round)
    np.save(os.path.join(args.root_log, args.store_name, 'test_avg_acc.npy'),calc_acc_ave(test_label,test_h)[0])

    print('classification report', classification_report(test_label,test_h))

    print("test_Average_accuracy",calc_acc_ave(test_label,test_h))

    with open(os.path.join(args.root_log, args.store_name, 'log_test.txt'), "w") as test_f:
        print(f"dataset : {args.dataset}",file=test_f)
        print(f"gamma = {gamma}",file=test_f)
        print(f"seed = {args.seed}",file=test_f)
        print("===================Test===================",file=test_f)
    
    with open(os.path.join(args.root_log, args.store_name, 'log_test.txt'), "a") as test_f:
        print('classification report', classification_report(test_label,test_h),file=test_f)
        print("Test_Average_accuracy",calc_acc_ave(test_label,test_h),file=test_f)

    print("--------------------------------------Finish---------------------------------------------------")
