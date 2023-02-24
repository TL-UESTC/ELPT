import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from matrix import *


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise NotImplementedError
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise NotImplementedError
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9*dsize)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders, len(dsets['target'])


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc


def train_target(args):
    dset_loaders, data_len = data_load(args)
    interval = 4
    sample_time = 5
    budget = 0.05
    init, r, label_flag = 0.75, 0, 1
    end_init = args.end
    step = (end_init-init) / (sample_time-1)
    P = [init+i*step for i in range(sample_time)]
    label_count_left = round(data_len * budget) 
    label_count_each = int(label_count_left * (1/sample_time)) + 1 
    interval_iter = 1*len(dset_loaders['target'])
    phase1_len = args.phase1 * interval_iter
    increase_km = int(0.6*args.phase1*interval_iter)
    max_iter = phase1_len + interval * (sample_time+1) * interval_iter
    print("interval_iter={}, max_iter={}; km={}, phase1={}".format(interval_iter, max_iter, increase_km, phase1_len))
    iter_num = -1
    inputs_next, pred_next, ready_flag, idx = [], [], 0, []
    p = 0
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256).cuda()
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    eng_bank = torch.randn(num_sample)
    sim_bank = torch.zeros(num_sample)
    args.ps = 0
    pre_iter = -9999
    knn = 1
    thre = 0

    print("loading model...")
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    print("load complete")

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    #building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,256)
    score_bank = torch.randn(num_sample, 12).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    print("initializing bank...")
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs = netC(output)
            energy = -torch.logsumexp(outputs, 1)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()
            eng_bank[indx] = energy.detach().clone().cpu()
    print("initialization complete")
    acc_log = 0

    while iter_num < max_iter:
        iter_num += 1
        if args.ps:
            if iter_num > pre_iter + interval*interval_iter and r < sample_time: 
                knn = 0
                label_flag = 1
            else:
                #knn = 1
                label_flag = 0

        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        netF.train()
        netB.train()
        netC.train()

        inputs_test = inputs_test.cuda()
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        if knn:
            fea_bank, score_bank, eng_bank, sim_bank = train_knn\
                (inputs_test, netF, netC, netB, fea_bank, score_bank, eng_bank, tar_idx, args, optimizer, sim_bank)

        if args.ps:
        ## obtain pseudo label
            if label_flag:
                pre_iter = iter_num
                label_cnt = label_count_left if label_count_each >= label_count_left else label_count_each
                netF.eval()
                netB.eval()
                netC.eval()
    
                p = P[r] if r<len(P) else P[-1]

                r += 1
                mem_label, idx, label_cnt, thre = obtain_label(dset_loaders['test'], netF, netB, netC, args, label_cnt, p, 0, sim_bank)
                netF.train()
                netC.train()
                netB.train()
                label_count_left -= label_cnt
                mem_label = torch.from_numpy(mem_label).cuda()

            inputs = inputs_test.cuda()
            aval_idx = [i for i in tar_idx if i in idx]
            if len(aval_idx) < 1:
                continue
            aval_idx = torch.tensor(aval_idx)
            mask = [True if tar_idx[i] in aval_idx else False for i in range(inputs.size(0))]
            inputs_tmp = inputs[mask]
            pred_tmp = mem_label[aval_idx]

            inputs_test, inputs_next, pred, pred_next, ready_flag = collect_data(inputs_next, inputs_tmp, pred_next, pred_tmp, args)
            if not ready_flag:
                label_flag = 0
            else:
                label_flag = 1
                for repeat in range(1):
                    features_test = netB(netF(inputs_test))
                    outputs = netC(features_test)
                    classifier_loss = nn.CrossEntropyLoss()(outputs, pred)
                    optimizer.zero_grad()
                    optimizer_c.zero_grad()
                    classifier_loss.backward()
                    optimizer.step()
                    optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + 'T: ' + acc_list

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
            best_netB = netB.state_dict()

        if iter_num == phase1_len :
            print("*"*40+"Phase 1 complete, starting labeling..."+"*"*40)
            args.ps = 1
            knn = 0
                
    torch.save(best_netF, osp.join(args.output_dir, "target_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "target_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "target_C.pt"))
    print('COMPLETE')


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--phase1', type=int, default=2, help="max epoch for phase 1")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--worker', type=int, default=4,)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--M', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='weight/', help='output path for DA task')
    parser.add_argument('--output_src', type=str, default='weight/', help='path to source_only weights')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--end', default=0.5, type=float)
    parser.add_argument('--ps', default=0, type=int)
    parser.add_argument('--ran', default=0, type=int)
    args = parser.parse_args()
    print("="*100)
    print("PS={}, end={}, K={}, M={}, ran={}".\
        format(args.ps, args.end, args.K, args.M, args.ran))

    names = ['train', 'validation']
    args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + names[args.s] + '_list.txt'
        args.t_dset_path = folder + names[args.t] + '_list.txt'
        args.test_dset_path = folder + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, names[args.s])
        args.output_dir = osp.join(args.output, args.da, names[args.s] + '2' + names[args.t])
        args.name = names[args.s] + '2' + names[args.t]

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log_target.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
