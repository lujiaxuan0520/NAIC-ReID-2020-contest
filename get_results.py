##########################################################
# read the saved model and get the json results
# Author: Jiaxuan Lu
# Example:
# python get_results.py -d pclreid \
#    -a resnet50 \
#    -j 0 \
#    --test-batch 100 \
#    --gpu-devices 5 \
#    --model-weight ./log/resnet50-xent/baseline_model.pth.tar \
#    --save-json ./baseline_submit.json
##########################################################
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import time
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchreid import data_manager, metrics
from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.torchtools import count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.utils.re_ranking import re_ranking
# from torchreid.utils.reranking import re_ranking

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')

parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    choices=models.get_names())
parser.add_argument('--global-branch', action='store_true',
                    help="whether to use the global branch in the architecture")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--test-batch', default=100, type=int,
                    help="test batch size")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--use-avai-gpus', action='store_true',
                    help="use available gpus instead of specified devices (this is useful when using managed clusters)")
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--root', type=str, default='./',
                    help="root path to data directory")
parser.add_argument('--model-weight', type=str, default='',
                    help="load the weights of trained model")
parser.add_argument('--save-json', type=str, default='',
                    help="the saved json path of the results")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index (0-based)")
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--dist-metric', type=str, default='euclidean',
                    help='distance metric')
parser.add_argument('--re-rank', action='store_true',
                    help='enable re-ranking in the testing stage.')
parser.add_argument('--vis-ranked-res', action='store_true',
                    help='visualize the ranked result or not.')


args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id, isFinal=True,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Normalize(mean=[0.3495, 0.3453, 0.3941], std=[0.2755, 0.2122, 0.2563]),
    ])

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test, isFinal=True),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test, isFinal=True),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    print("Initializing model: {}".format(args.arch))
    '''
       vmgn_hgnn model, arch chosen from {'resnet50','resnet101','resnet152'}
       efficientnet_hgnn model, arch chosen from {'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
       'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7','efficientnet-b8'}
       '''
    model = models.init_model(name=args.arch,
                              # num_classes=29626, # 30874 or 20906 or 29626 or 34394
                              num_classes=19658,
                              isFinal=True,
                              global_branch=args.global_branch,
                              arch="resnet101")
    print("Model size: {:.3f} M".format(count_num_param(model)))

    checkpoint = torch.load(args.model_weight)
    pretrain_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()
    # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # model_dict.update(pretrain_dict)
    model.load_state_dict(pretrain_dict)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print("Evaluate only")
    distmat, q_img_paths, g_img_paths = test_final(model, queryloader, galleryloader, use_gpu)

    # save the distmap, q_img_paths, g_img_paths for further ensemble
    file_name = args.save_json.replace("./", "./results/").replace(".json",".npy")
    np.save(file_name, distmat)
    np.save(file_name.replace(".npy", "_q_img_paths.npy"), q_img_paths)
    np.save(file_name.replace(".npy", "_g_img_paths.npy"), g_img_paths)

    res_dict = dict()
    for query_idx, line in enumerate(distmat):
        query_name = q_img_paths[query_idx]
        gallery_top_200_idx = np.argsort(line)[:200] # the index of the top 200 similiar images in gallery
        gallery_top_200_name = [g_img_paths[item] for item in gallery_top_200_idx]
        res_dict[query_name] = gallery_top_200_name

    # save the json results
    json_str = json.dumps(res_dict)
    with open(args.save_json, 'w') as json_file:
        json_file.write(json_str)

    if args.vis_ranked_res:
        visualize_ranked_results(
            distmat, dataset,
            save_dir=args.save_json.replace("./","./img/").replace(".json",""),
            topk=20,
        )

    print("Done.")



def test_final(model, queryloader, galleryloader, use_gpu):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_img_paths, q_camids = [], [], []
        for batch_idx, (imgs,img_paths, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            img_paths = [item.split('/')[-1] for item in img_paths]
            q_img_paths.extend(img_paths)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        # q_img_paths = np.asarray(q_img_paths)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_img_paths, g_camids = [], [], []
        for batch_idx, (imgs, img_paths, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            img_paths = [item.split('/')[-1] for item in img_paths]
            g_img_paths.extend(img_paths)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        # g_img_paths = np.asarray(g_img_paths)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    # m, n = qf.size(0), gf.size(0)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())

    # slow re-ranking
    distmat = metrics.compute_distance_matrix(qf, gf, args.dist_metric)
    distmat = distmat.numpy()
    if args.re_rank:
        print('Applying person re-ranking ...')
        distmat_qq = metrics.compute_distance_matrix(qf, qf, args.dist_metric)
        distmat_gg = metrics.compute_distance_matrix(gf, gf, args.dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg, k1=20, k2=6, lambda_value=0.3) # default: (20,6,0.3)

    # fast re-ranking
    # if args.re_rank:
    #     print('Applying person re-ranking ...')
    #     # distmat_qq = metrics.compute_distance_matrix(qf, qf, args.dist_metric)
    #     # distmat_gg = metrics.compute_distance_matrix(gf, gf, args.dist_metric)
    #     # distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    #     distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    # else:
    #     distmat = metrics.compute_distance_matrix(qf, gf, args.dist_metric)
    #     distmat = distmat.numpy()

    return distmat, q_img_paths, g_img_paths


main()
# if __name__ == '__main__':
#     main()