from utils import * 
from datasets import CustomDataset 


def score(args,loader,model,round,model_path,device):
    # args.store_name = '_'.join([str(args.min_size), str(args.test_min_size), str(args.seed)])
    ht = []
    prepare_folders(args)

    model = model
    test_loader = loader
    # check_datasets = AreaDataset()
    # check_dataloader = DataLoader(check_datasets, batch_size=batch, shuffle=False) #generator=torch.Generator().manual_seed(42)
    for t in tqdm(range(round)):
        model.load_state_dict(torch.load(model_path +f'/weak-l({t}).pt'))
        model  = model.to(device)
        _,tmp_h = class_wise_acc(model,test_loader,device)
        ht.append(tmp_h)
    ht = transposition(ht)
    h = voting(ht)

    test_labels,_,_ = make_label_data_clsnum(test_loader)

    test_acc = calc_acc(test_labels,h)

    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    out_cls_acc = 'Class Accuracy: %s'%(np.array2string(np.array(test_acc), separator=',', 
                                                        formatter={'float_kind':lambda x: "%.4f" % x}))
    log_testing.write(out_cls_acc + '\n')
    log_testing.flush()

    return True






 