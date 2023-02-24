import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import random


already_labeled_idx = []

def KL(p, q):
    '''
    p,q: shape(btz, x, C), C be probability distribution
    return: shape(btz, x)
    '''
    return (p*torch.log(p/q)).sum(-1)


def obtain_label(loader, netF, netB, netC, args, label_cnt=0, percen=0.5, last=0, sim_bank=[]):
    #print("percen={}".format(percen))
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            outputs_eng = -torch.logsumexp(outputs, 1)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_eng = outputs_eng.cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_eng = torch.cat((all_eng, outputs_eng.cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    # sort samples according to ENERGY, then decide thre_a
    eng_list = all_eng.tolist()
    idx_list = list(range(len(eng_list)))
    sim_list = sim_bank
    #print(len(eng_list), len(idx_list))
    z = zip(eng_list, idx_list, sim_list)
    sort_list = sorted(z, key=lambda x:(x[0]), reverse=True)
    #print(sort_list)
    sorted_eng_list, sorted_idx_list, sorted_sim_list = zip(*sort_list)
    l = len(sorted_idx_list)

    # first choose top 5% data with largest energy
    sorted_idx_list = sorted_idx_list[:int(0.07*l)]
    sorted_sim_list = sorted_sim_list[:int(0.07*l)]

    # then sort according to feature similarity and choose those with less similarity 
    z = zip(sorted_sim_list, sorted_idx_list)
    sort_list = sorted(z, key=lambda x:(x[0]), reverse=False)
    sorted_sim_list, sorted_idx_list = zip(*sort_list)

    ori_sorted_idx_list = sorted_idx_list
    thre_a = sorted_eng_list[-int(l*percen)]
    thre_w = sorted_eng_list[int(l*0.2)]
    acc_rate = torch.sum(torch.squeeze(predict)[all_eng<thre_a].float() == all_label[all_eng<thre_a]).item() / float(all_label[all_eng<thre_a].size()[0])*100
    acc_num = all_label[all_eng<thre_a].size()[0]
    unk_rate = torch.sum(torch.squeeze(predict)[all_eng>thre_w].float() == all_label[all_eng>thre_w]).item() / float(all_label[all_eng>thre_w].size()[0])*100
    unk_num = all_label[all_eng>thre_w].size()[0]
    log_str = "acc count = {}, acc rate = {:.3f} %".format(acc_num, acc_rate)

    # label samples with HIGHEST energy && previously unlabeled
    sorted_idx_list = []
    selected_cnt, cur_idx, pre_lbl = 0, 0, 0
    if not args.ran:
        while selected_cnt < label_cnt and cur_idx < len(ori_sorted_idx_list):
            now = ori_sorted_idx_list[cur_idx]
            if now not in already_labeled_idx:
                sorted_idx_list.append(now)
                already_labeled_idx.append(now)
                selected_cnt += 1
            else:
                pre_lbl += 1  # previously labeled
            cur_idx += 1
    else:
        # randomly select
        while selected_cnt < label_cnt:
            idx = random.choice(idx_list)
            if idx not in sorted_idx_list:
                sorted_idx_list.append(idx)
                already_labeled_idx.append(idx)
                selected_cnt += 1
    
    eng_weight = torch.ones(len(all_eng)).unsqueeze(1)
    eng_weight = np.array(eng_weight)
    for i in range(len(eng_weight)):
        if all_eng[i] > thre_w:
            eng_weight[i] = 0.1
    all_fea = all_fea * eng_weight
    pred_label, _ = clustering(aff, all_fea, K, predict)
    acc_1 = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    print("///////////// CLUSTERING ACC = {:.3f} % ////////////////".format(acc_1*100))

    labeled_cnt = 0
    if label_cnt > 0:
        ori_acc_cnt = 0
        print("Labeled {} samples".format(len(sorted_idx_list)))
        labeled_cnt = len(sorted_idx_list)
        for i in sorted_idx_list:
            if pred_label[i] == all_label[i]:
                ori_acc_cnt += 1
        print("Original clustering acc rate for selected samples: {:.2f} % (the lower the better)".\
            format(ori_acc_cnt / len(sorted_idx_list) * 100))

    for i in already_labeled_idx:
        pred_label[i] = all_label[i]
        all_eng[i] = -100000   # satisfy 'all_eng < -thre_a'

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    ood_idx = np.where(all_eng >= thre_w)[0]
    pred_idx = np.where(all_eng < thre_a)[0]
    if last:
        pred_idx = np.where(all_eng < 0)[0] 
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print("Returned labels accuracy = {:.3f} %".format(np.sum(pred_label[pred_idx]==np.array(all_label[pred_idx]))/len(pred_idx)*100))
    #print(log_str+'\n')
    return pred_label.astype('int'), torch.tensor(pred_idx), labeled_cnt, thre_w
   

def clustering(aff, all_fea, K, predict):
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]  
    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    return pred_label, dd


def collect_data(inputs_test, inputs_new, pred, pred_new, args):
    '''
    returns: inputs_test, inputs_next, ready_flag
    '''
    if len(inputs_test)+len(inputs_new) >= args.batch_size:
        if len(inputs_test) == 0:
            inputs_test = inputs_new
            pred = pred_new
            inputs_next = []
            pred_next = []
        else:
            split = args.batch_size - len(inputs_test)
            inputs_test = torch.cat((inputs_test, inputs_new[:split]), 0)
            pred = torch.cat((pred, pred_new[:split]), 0)
            if split == 0:
                inputs_next = []
                pred_next = []
            else:
                inputs_next = inputs_new[split:]
                pred_next = pred_new[split:]
        return inputs_test, inputs_next, pred, pred_next, 1
    else:
        if len(inputs_test) == 0:
            inputs_next = inputs_new
            pred_next = pred_new
        else:
            inputs_next = torch.cat((inputs_test, inputs_new), 0)
            pred_next = torch.cat((pred, pred_new), 0)
        return [], inputs_next, [], pred_next, 0


def train_knn(inputs_test, netF, netC, netB, fea_bank, score_bank, eng_bank, tar_idx, args, optimizer, sim_bank):
    inputs_target = inputs_test.cuda()
    features_test = netB(netF(inputs_target))
    output = netC(features_test)
    energy = -torch.logsumexp(output, 1)
    softmax_out = nn.Softmax(dim=1)(output)
    btz = len(inputs_target)
    with torch.no_grad():
        output_f_norm = F.normalize(features_test)      
        output_f_ = output_f_norm.detach().clone()
        fea_bank[tar_idx] = output_f_.detach().clone().cpu()
        score_bank[tar_idx] = softmax_out.detach().clone()
        eng_bank[tar_idx] = energy.detach().clone().cpu()
        w = torch.matmul(output_f_norm.cpu(), fea_bank.t())  # compute adjacent matrix

        # k-nearest
        dist_k, idx_knear = torch.topk(w, dim=1, largest=True, k=args.K+1)  # shape(btz, K+1)
        idx_knear = idx_knear[:, 1:]  # shape(btz, K)
        dist_k = dist_k[:, 1:]
        sim_bank[tar_idx] = dist_k.mean(1).cpu()  # update simbank
        weight_k = torch.clamp(torch.exp(dist_k) - 1, min=0.1, max=1)
        score_near_k = score_bank[idx_knear].cuda()  # shape(btz, K, class_num)
        #print(weight_k)

        # m-nearest of each k
        fea_norm = fea_bank[idx_knear].cpu()  # shape(btz, K, dim)
        fea_bank_m = fea_bank.unsqueeze(0).expand(btz, -1, -1).permute(0, 2, 1)  # shape(btz, dim, n) 
        w = torch.bmm(fea_norm, fea_bank_m)   # compute adjacent matrix
        dist_m, idx_mnear = torch.topk(w, dim=2, largest=True, k=args.M+1)  # shape(btz, K, M+1)
        idx_mnear = idx_mnear[:, :, 1:]  # shape(btz, K, M)
        dist_m = dist_m[:, :, 1:]
        idx_mnear = idx_mnear.contiguous().view(btz, -1) # shape(btz, K*M)
        dist_m = dist_m.contiguous().view(btz, -1)
        weight_m = torch.ones_like(dist_m).fill_(0.1)
        score_near_m = score_bank[idx_mnear].cuda() # shape(btz, K*M, class_num)

    # train
    out_k = softmax_out.unsqueeze(1).expand(-1, args.K, -1)  #shape(btz, K, C)
    div_k = (- out_k * score_near_k).sum(-1) * weight_k.cuda()
    loss = torch.mean(div_k.sum(1))

    out_m = softmax_out.unsqueeze(1).expand(-1, args.K*args.M, -1)
    div_m = (- out_m * score_near_m).sum(-1) * weight_m.cuda()
    loss += torch.mean(div_m.sum(1))

    msoftmax = softmax_out.mean(dim=0)
    im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-6))
    loss += im_div  

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return fea_bank, score_bank, eng_bank, sim_bank


if __name__ == "__main__":
    pass