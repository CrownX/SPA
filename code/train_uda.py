import argparse
import random
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import loss
import utils


def gauss_(v1, v2, sigma):
    norm_ = torch.norm(v1 - v2, p=2, dim=0)
    return torch.exp(-0.5 * norm_ / sigma**2)

def euc_(v1, v2):
    return torch.norm(v1 - v2, p=2, dim=0)

def adj_(s, t, ap='cos'):
    # s, t [bsize, dim], [bsize, dim] -> [bsize, bsize]
    if ap == 'cos':
        s_norm = s / torch.norm(s, p=2, dim=1, keepdim=True)
        t_norm = t / torch.norm(t, p=2, dim=1, keepdim=True)
        adj = s_norm @ t_norm.T
    elif ap == 'gauss':
        sigma_ = 1.5
        M, N = s.shape[0], t.shape[0]
        adj = torch.zeros([M, N], dtype=torch.float).cuda()
        for i in range(M):
            for j in range(i):
                adj[i][j] = adj[j][i] = gauss_(s[i], t[j], sigma_)
    elif ap == 'euc':
        M, N = s.shape[0], t.shape[0]
        adj = torch.zeros([M, N], dtype=torch.float).cuda()
        for i in range(M):
            for j in range(i):
                adj[i][j] = adj[j][i] = euc_(s[i], t[j])
    return adj
    
def laplacian_(A, ltype='laplac1'):
    v = torch.sum(A, dim=1)
    if ltype == 'laplac1':
        v_inv = 1 / v
        D_inv = torch.diag(v_inv).cuda()
        return -D_inv @ A
    elif ltype == 'laplac2':
        D = torch.diag(v).cuda()
        return D - A
    elif ltype == 'laplac3':
        v_sqrt = 1 / torch.sqrt(v)
        D_sqrt = torch.diag(v_sqrt).cuda()
        I = torch.eye(A.shape[0]).cuda()
        return I - D_sqrt @ A @ D_sqrt

def svd_loss_(s, t):
    # s, t [bsize, dim], [bsize, dim]
    s_matrix = adj_(s, s, args.ap)
    t_matrix = adj_(t, t, args.ap)
    s_matrix = laplacian_(s_matrix, args.laplac)
    t_matrix = laplacian_(t_matrix, args.laplac)
    _, s_v, _ = torch.svd(s_matrix)
    _, t_v, _ = torch.svd(t_matrix)
    svd_loss = torch.norm(s_v - t_v, p=2)
    return svd_loss

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def data_load(args, labels=None):
    train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomCrop((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_set = utils.ObjectImage(args.data_root, args.s_dset_path, train_transform, y=labels)
    target_set = utils.ObjectImage(args.data_root, args.t_dset_path, train_transform, ridx=True)
    test_set = utils.ObjectImage(args.data_root, args.test_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders


def train(args, validate=False, label=None):
    ## set pre-process
    dset_loaders = data_load(args, label)
    class_num = args.class_num
    class_weight_src = torch.ones(class_num, ).cuda()
    ##################################################################################################
    ## set base network
    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()  

    netF = utils.ResClassifier(class_num=class_num, feature_dim=netG.in_features, 
        bottleneck_dim=args.bottleneck_dim).cuda()

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len

    ad_flag = False
    if args.method in {'DANN', 'DANNE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim, 1024, max_iter=args.max_iter).cuda()
        ad_flag = True
    if args.method in {'CDAN', 'CDANE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim*class_num, 1024, max_iter=args.max_iter).cuda() 
        random_layer = None
        ad_flag = True  

    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr * 0.1)
    optimizer_f = optim.SGD(netF.parameters(), lr = args.lr)
    if ad_flag:
        optimizer_d = optim.SGD(ad_net.parameters(), lr = args.lr)
   
    base_network = nn.Sequential(netG, netF)

    mem_fea = torch.rand(len(dset_loaders["target"].dataset), args.bottleneck_dim).cuda()
    mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
    mem_cls = torch.ones(len(dset_loaders["target"].dataset), class_num).cuda() / class_num

    source_loader_iter = iter(dset_loaders["source"])
    target_loader_iter = iter(dset_loaders["target"])
    ####
    list_acc = []
    best_ent = 100
    for iter_num in range(1, args.max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        if ad_flag:
            lr_scheduler(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = source_loader_iter.next()
        try:
            inputs_target, _, idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target, _, idx = target_loader_iter.next()
        
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        if args.method == 'srconly' and args.pl == 'none':
            features_source, outputs_source = base_network(inputs_source)
        else:
            features_source, outputs_source = base_network(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

        eff = utils.calc_coeff(iter_num, max_iter=args.max_iter)
        if args.method[-1] == 'E':
            entropy = loss.Entropy(softmax_out)
        else:
            entropy = None

        if args.method in {'CDAN', 'CDANE'}:           
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, eff, random_layer)
        elif args.method in {'DANN', 'DANNE'}:  
            transfer_loss = loss.DANN(features, ad_net, entropy, eff)
        elif args.method == 'srconly':
            transfer_loss = torch.tensor(0.0).cuda()
        else:
            raise ValueError('Method cannot be recognized.')

        src_ = loss.CrossEntropyLabelSmooth(reduction='none',num_classes=class_num, epsilon=args.smooth)(outputs_source, labels_source)
        weight_src = class_weight_src[labels_source].unsqueeze(0)
        classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
        total_loss = transfer_loss + classifier_loss

        eff = iter_num / args.max_iter

        if args.ifcorrect:
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
        dis = -torch.mm(features_target.detach(), mem_fea.t())
        for di in range(dis.size(0)):
            dis[di, idx[di]] = torch.max(dis)
        _, p1 = torch.sort(dis, dim=1)

        w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w[wi][p1[wi, wj]] = 1/ args.K
        weight_, pred = torch.max(w.mm(mem_cls), 1)

        loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_target, pred)
        classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())   
        pl_loss = args.tar_par * eff * classifier_loss
        if args.pl != 'none':
            total_loss += pl_loss

        if args.ifsvd:
            # svd loss
            f_s = features_source
            f_t = features_target
            # svd_loss = args.svd_par * eff * svd_loss_(f_s, f_t)
            svd_loss = args.svd_par * svd_loss_(f_s, f_t)
            total_loss += svd_loss
            
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if ad_flag:
            optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        if ad_flag:
            optimizer_d.step()

        base_network.eval() 
        with torch.no_grad():
            features_target, outputs_target = base_network(inputs_target)
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))

            mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
            mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()


        if iter_num % 10 == 0:
            iter_str = 'total:{:.5f}, trans: {:.5f}, cls: {:.5f}'.format(total_loss.item(), transfer_loss.item(), classifier_loss.item())
            if args.pl != 'none':
                iter_str += ', pl:{:.5f}'.format(pl_loss.item())
            if args.ifsvd:
                iter_str += ', svd:{:.5f}'.format(svd_loss.item())
            print(iter_str)
            
        if iter_num % int(max_len) == 0:
            base_network.eval()
            if args.dset == 'visda2017':
                acc, py, score, y, tacc = utils.cal_acc_visda(dset_loaders["test"], base_network)
                args.out_file.write(tacc + '\n')
                args.out_file.flush()
                print(tacc)

                _ent = loss.Entropy(score)
                mean_ent = 0
                for ci in range(args.class_num):
                    mean_ent += _ent[py==ci].mean()
                mean_ent /= args.class_num
            else:
                acc, py, score, y = utils.cal_acc(dset_loaders["test"], base_network)
                mean_ent = torch.mean(loss.Entropy(score))

            list_acc.append(acc * 100)
            if best_ent > mean_ent:
                best_ent = mean_ent
                val_acc = acc * 100
                best_y = y
                best_py = py
                best_score = score

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, iter_num, args.max_iter, acc*100, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')            

    idx = np.argmax(np.array(list_acc))
    max_acc = list_acc[idx]
    final_acc = list_acc[-1]

    log_str = '\n==========================================\n'
    log_str += '\nVal Acc = {:.2f}\nMax Acc = {:.2f}\nFin Acc = {:.2f}\n'.format(val_acc, max_acc, final_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')  
    
    return best_y.cpu().numpy().astype(np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='srconly', choices=['srconly', 'CDAN', 'CDANE', 'DANN', 'DANNE'])
    parser.add_argument('--pl', type=str, default='none', choices=['none', 'spa', 'npl', 'bsp'])

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--tar_par', type=float, default=1.0)
    parser.add_argument('--validate', action='store_true')
    
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet50", "resnet101"])
    parser.add_argument('--dset', type=str, default='office_home', choices=['domain_net', 'multi', 'visda2017', 'office31', 'office_home'], help="dataset used")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    parser.add_argument('--ifcorrect', action='store_true')
    parser.add_argument('--ifsvd', action='store_true')
    parser.add_argument('--svd_par', type=float, default=1.0)
    parser.add_argument('--laplac', type=str, default='laplac1', choices=["laplac1", "laplac2", "laplac3"])
    parser.add_argument('--ap', type=str, default='euc', choices=['cos', 'gauss', 'euc'])
    args = parser.parse_args()
    args.output = args.output.strip()

    args.eval_epoch = args.max_epoch / 10

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65 
        args.data_root = ''
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        args.data_root = ''
    if args.dset == 'visda2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'multi': # DomainNet-126
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
        args.data_root = '/data/domain_net/'
    if args.dset == 'domain_net':
        names = ['clipart_train', 'painting_train', 'real_train', 'sketch_train']
        tests = ['clipart_test', 'painting_test', 'real_test', 'sketch_test']
        args.class_num = 345  
        args.data_root = ''

    args.s_dset_path = './data/uda/' + args.dset + '/' + names[args.s] + '.txt'
    args.t_dset_path = './data/uda/' + args.dset + '/' + names[args.t] + '.txt'
    if args.dset == 'domain_net':
        args.test_dset_path = './data/' + args.dset + '/' + tests[args.t] + '.txt'
    else:  
        args.test_dset_path = args.t_dset_path

    if args.pl == 'none':
        args.output_dir = osp.join(args.output, args.pl, args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())
    else:
        args.output_dir = osp.join(args.output, args.pl + '_' + str(args.tar_par), args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.log = args.method
    args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    utils.print_args(args)
    label = train(args)
    if args.validate:
        train(args, validate=True, label=label)