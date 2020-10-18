#########################################################################################
# semi-supervised learning: use label propagation to make pseudo labels for no label data
# This is not the parallel implement of label propagation, may requires a lot of time
# Author: Jiaxuan Lu
#########################################################################################


import time
import numpy as np
import math
import os, sys
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.torchtools import count_num_param

gpu_devices = "7" # gpu devices
extended_data = False # whether to use extended data
model_weight = "./log/resnet50-xent/vmgn_hgnn13/checkpoint_ep65.pth.tar"
arch = "vmgn_hgnn"
test_batch = 500
dataset_name = "pclreid"
global_branch = True
dist_metric = "cosine"
root = "./"
height = 256
width = 128
seed = 1
workers = 4

# return k neighbors index
def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]

    ## step 1: calculate Euclidean distance
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row

    ## step 2: sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)

    return sortedDistIndices[0:k]


# build a big graph (normalized weight matrix)
def buildGraph(MatX, kernel_type, rbf_sigma=None, knn_num_neighbors=None):
    num_samples = MatX.shape[0]
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)
    if kernel_type == 'rbf':
        if rbf_sigma == None:
            raise ValueError('You should input a sigma of rbf kernel!')
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i][j] = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == 'knn':
        if knn_num_neighbors == None:
            raise ValueError('You should input a k of knn kernel!')
        for i in range(num_samples):
            k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError('Not support kernel type! You can use knn or rbf!')

    return affinity_matrix


# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='rbf', rbf_sigma=1.5, \
                     knn_num_neighbors=10, max_iter=500, tol=1e-3):
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)

    MatX = np.vstack((Mat_Label, Mat_Unlabel))
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0

    label_function = np.zeros((num_samples, num_classes), np.float32)
    label_function[0: num_label_samples] = clamp_data_label
    label_function[num_label_samples: num_samples] = -1

    # graph construction
    print("start build graph.")
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)
    print("build graph done.")

    # start to propagation
    iter = 0
    pre_label_function = np.zeros((num_samples, num_classes), np.float32)
    changed = np.abs(pre_label_function - label_function).sum()
    while iter < max_iter and changed > tol:
        if iter % 1 == 0:
            print("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))
        pre_label_function = label_function
        iter += 1

        # propagation
        label_function = np.dot(affinity_matrix, label_function)

        # clamp
        label_function[0: num_label_samples] = clamp_data_label

        # check converge
        changed = np.abs(pre_label_function - label_function).sum()

    # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i + num_label_samples])

    return unlabel_data_labels


def process_dir_label(list_path,cam):
    with open(list_path, 'r') as txt:
        lines = txt.readlines()
    dataset = []
    pid_container = set()
    for img_idx, img_info in enumerate(lines):

        img_path, pid = img_info.split(':')
        pid = int(pid) # no need to relabel

        camid = cam
        # img_path = osp.join(dir_path, img_path)
        dataset.append((img_path, pid, camid))
        pid_container.add(pid)
    num_imgs = len(dataset)
    num_pids = len(pid_container)
    # check if pid starts from 0 and increments with 1

    return dataset, num_pids, num_imgs

def process_dir_unlabel(list_path,cam):
    with open(list_path, 'r') as txt:
        lines = txt.readlines()
    dataset = []
    for img_idx, img_info in enumerate(lines):

        img_path = img_info.replace("\n","")
        camid = cam
        # img_path = osp.join(dir_path, img_path)
        dataset.append((img_path, camid))
    num_imgs = len(dataset)

    return dataset, num_imgs

def test(model, labelloader, unlabelloader, use_gpu):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        label_feature, label_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(labelloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            label_feature.append(features)
            label_pids.extend(pids)
            # q_camids.extend(camids)
        label_feature = torch.cat(label_feature, 0)
        label_pids = np.asarray(label_pids)
        # q_camids = np.asarray(q_camids)

        unlabel_feature, unlabel_img_path, g_camids = [], [], []
        for batch_idx, (imgs, img_path, camids) in enumerate(unlabelloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            unlabel_feature.append(features)
            unlabel_img_path.extend(img_path)
            # g_camids.extend(camids)
        unlabel_feature = torch.cat(unlabel_feature, 0)
        # unlabel_img_path = np.asarray(unlabel_img_path)
        # g_camids = np.asarray(g_camids)

    return label_feature, unlabel_feature, label_pids, unlabel_img_path


# main function
if __name__ == "__main__":
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print("Currently using GPU {}".format(gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(dataset_name))

    dataset_dir = osp.join(root, 'PCL_ReID')
    list_label_path = osp.join(dataset_dir, 'train_extended_list.txt') if extended_data else \
        osp.join(dataset_dir, 'train_list.txt')
    list_unlabel_path = osp.join(dataset_dir, 'no_label_extended_list.txt') if extended_data else \
        osp.join(dataset_dir, 'no_label_list.txt')

    label_data, num_label_pids, num_label_imgs = process_dir_label(list_label_path, cam=0)
    unlabel_data, num_unlabel_imgs = process_dir_unlabel(list_unlabel_path, cam=1)

    transform_test = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Normalize(mean=[0.3495, 0.3453, 0.3941], std=[0.2755, 0.2122, 0.2563]),
    ])

    pin_memory = True if use_gpu else False

    labelloader = DataLoader(
        ImageDataset(label_data, transform=transform_test),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False
    )

    unlabelloader = DataLoader(
        ImageDataset(unlabel_data, transform=transform_test, isFinal=True),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False
    )

    print("Initializing model: {}".format(arch))

    '''
           vmgn_hgnn model, arch chosen from {'resnet50','resnet101','resnet152'}
           efficientnet_hgnn model, arch chosen from {'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
           'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7','efficientnet-b8'}
           '''
    model = models.init_model(name=arch,
                              num_classes=29626, # 29626 or 34394
                              # num_classes=19658,
                              isFinal=False,
                              global_branch=global_branch,
                              arch="resnet101")
    print("Model size: {:.3f} M".format(count_num_param(model)))

    checkpoint = torch.load(model_weight)
    pretrain_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()
    # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # model_dict.update(pretrain_dict)
    model.load_state_dict(pretrain_dict)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print("Evaluate only")
    Mat_Label, Mat_Unlabel, labels, unlabel_img_path = test(model, labelloader, unlabelloader, use_gpu)

    # num_unlabel_samples = 800
    # Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)
    # Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)

    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    # unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)
    print("start label propagation")
    unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='knn', knn_num_neighbors=5,
                                           max_iter=400)
    # show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)

    for idx in range(len(unlabel_img_path)):
        unlabel_img_path[idx] += ':' + str(unlabel_data_labels[idx])

    np.savetxt("pseudo_label_for_no_label.txt", unlabel_img_path)