from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchreid import data_manager
from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import set_bn_to_eval, count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optim
#import models.resnet_person

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index (0-based)")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=100, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--fixbase-epoch', default=0, type=int,
                    help="epochs to fix base network (only train classifier, default: 0)")
parser.add_argument('--fixbase-lr', default=0.0003, type=float,
                    help="learning rate (when base network is frozen)")
parser.add_argument('--freeze-bn', action='store_true',
                    help="freeze running statistics in BatchNorm layers during training (default: False)")
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--use-avai-gpus', action='store_true',
                    help="use available gpus instead of specified devices (this is useful when using managed clusters)")
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.3f} M".format(count_num_param(model)))


    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        if check_isfile(args.load_weights):
            checkpoint = torch.load(args.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))


    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        distmat = test(model,dataset, transform_test,queryloader, galleryloader, use_gpu, return_distmat=True)
        if args.vis_ranked_res:
            visualize_ranked_results(
                distmat, dataset,
                save_dir=osp.join(args.save_dir, 'ranked_results'),
                topk=20,
            )
        return


def test(model,dataset, transform_test,queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    

    from PIL import Image


    batch_time = AverageMeter()
    
    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []

        for (img_path, pid, camid) in dataset.query:

            img = Image.open(img_path).convert('RGB')
            transform=transform_test
            img = transform(img)

            imgs=img.view(1,3,256,128)
            pids=np.array([pid])
            camids=np.array([camid])
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)


        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for (img_path, pid, camid) in dataset.gallery:

            img = Image.open(img_path).convert('RGB')
            transform=transform_test
            img = transform(img)
            imgs=img.view(1,3,256,128)
            pids=np.array([pid])
            camids=np.array([camid])
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)


        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)



        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    '''
	find rank1 positional information
    '''
    #f = open('answer_a.txt','w')
    #f.write('\n'.join(str(i) for i in np.argmin(distmat,1).tolist()))
    #f.close()

    '''
	find rank200 positional information	
    '''
    #f = open('answer_b_200.txt','w')
    #indices = np.argsort(distmat,1)
    #indices_200 = indices[:,:200]
    #f.write('\n'.join(str(i) for i in indices_200.tolist()))
    #f.close()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
