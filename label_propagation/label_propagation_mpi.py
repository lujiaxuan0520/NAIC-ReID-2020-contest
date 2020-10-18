################################################################################################
# semi-supervised learning: use label propagation to make pseudo labels for no label data
# This is the parallel version of label propagation, required openmpi and Cython
# Author: Jiaxuan Lu
# run label_propagation_mpi_get_feature.py first, and then
#   run "mpirun -np 5 label_propagation_mpi.py", finally run label_propagation_postprocessing.py
################################################################################################


import time
import numpy as np
import math
import os, sys, time
import os.path as osp
from scipy.sparse import csr_matrix, lil_matrix, eye
import operator
import _pickle as pickle
import mpi4py.MPI as MPI


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.torchtools import count_num_param

Mat_Label_file = "./label_propagation/mat_label.npy"
Mat_Unlabel_file = "./label_propagation/mat_unlabel.npy"
labels_file = "./label_propagation/labels.npy"
unlabel_img_path_file = "./label_propagation/unlabel_img_path.npy"

# instance for invoking MPI related functions
comm = MPI.COMM_WORLD
# the node rank in the whole community
comm_rank = comm.Get_rank()
# the size of the whole community, i.e., the total number of working nodes in the MPI cluster
comm_size = comm.Get_size()


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
# sparse U x (U + L) matrix
def buildSubGraph(Mat_Label, Mat_Unlabel, knn_num_neighbors):
    num_unlabel_samples = Mat_Unlabel.shape[0]
    data = []
    indices = []
    indptr = [0]
    Mat_all = np.vstack((Mat_Label, Mat_Unlabel))
    values = np.ones(knn_num_neighbors, np.float32) / knn_num_neighbors
    for i in range(num_unlabel_samples):
        k_neighbors = navie_knn(Mat_all, Mat_Unlabel[i, :], knn_num_neighbors)
        indptr.append(np.int32(indptr[-1]) + knn_num_neighbors)
        indices.extend(k_neighbors)
        data.append(values)
    return csr_matrix((np.hstack(data), indices, indptr))


# build a big graph (normalized weight matrix)
# sparse U x (U + L) matrix
def buildSubGraph_MPI(Mat_Label, Mat_Unlabel, knn_num_neighbors):
    num_unlabel_samples = Mat_Unlabel.shape[0]
    local_data = []
    local_indices = []
    local_indptr = [0]
    Mat_all = np.vstack((Mat_Label, Mat_Unlabel))
    values = np.ones(knn_num_neighbors, np.float32) / knn_num_neighbors
    sample_offset = np.linspace(0, num_unlabel_samples, comm_size + 1).astype('int')
    for i in range(sample_offset[comm_rank], sample_offset[comm_rank + 1]):
        k_neighbors = navie_knn(Mat_all, Mat_Unlabel[i, :], knn_num_neighbors)
        local_indptr.append(np.int32(local_indptr[-1]) + knn_num_neighbors)
        local_indices.extend(k_neighbors)
        local_data.append(values)
    data = np.hstack(comm.allgather(local_data))
    indices = np.hstack(comm.allgather(local_indices))
    indptr_tmp = comm.allgather(local_indptr)
    indptr = []
    for i in range(len(indptr_tmp)):
        if i == 0:
            indptr.extend(indptr_tmp[i])
        else:
            last_indptr = indptr[-1]
            del (indptr[-1])
            indptr.extend(indptr_tmp[i] + last_indptr)
    return csr_matrix((np.hstack(data), indices, indptr), dtype=np.float32)


def evaluation(num_unlabel_samples, local_start_class, local_label_function_U, labels_id):
    # get local label with max score
    if comm_rank == 0:
        print("Start to combine local result...")
    local_max_score = np.zeros((num_unlabel_samples, 1), np.float32)
    local_max_label = np.zeros((num_unlabel_samples, 1), np.int32)
    for i in range(num_unlabel_samples):
        local_max_label[i, 0] = np.argmax(local_label_function_U.getrow(i).todense())
        local_max_score[i, 0] = local_label_function_U[i, local_max_label[i, 0]]
        local_max_label[i, 0] += local_start_class

    # gather the results from all the processors
    if comm_rank == 0:
        print("Start to gather results from all processors")
    all_max_label = np.hstack(comm.allgather(local_max_label))
    all_max_score = np.hstack(comm.allgather(local_max_score))

    unlabel_data_labels = np.zeros(num_unlabel_samples)
    # get terminate label of unlabeled data
    if comm_rank == 0:
        print("Start to analysis the results...")
        # right_predict_count = 0
        for i in range(num_unlabel_samples):
            if i % 1000 == 0:
                print("***", all_max_score[i])
            max_idx = np.argmax(all_max_score[i])
            max_label = all_max_label[i, max_idx]
            unlabel_data_labels[i] = labels_id[max_label]
        #     if int(unlabel_data_id[i]) == int(labels_id[max_label]):
        #         right_predict_count += 1
        # accuracy = float(right_predict_count) * 100.0 / num_unlabel_samples
    return unlabel_data_labels


# label propagation
def label_propagation_sparse(Mat_Label, Mat_Unlabel,labels, unlabel_img_path, kernel_type="knn", knn_num_neighbors=5, max_iter=100, tol=1e-4, test_per_iter=10):
    # Mat_Label, labels, Mat_Unlabel, groundtruth = loadFourBandData()
    # Mat_Label, labels, labels_id, Mat_Unlabel, unlabel_data_id = load_MNIST()
    labels_id = np.unique(labels)
    if comm_size > len(labels_id):
        raise ValueError("Sorry, the processors must be less than the number of classes")
    # affinity_matrix = buildSubGraph(Mat_Label, Mat_Unlabel, knn_num_neighbors)
    affinity_matrix = buildSubGraph_MPI(Mat_Label, Mat_Unlabel, knn_num_neighbors)

    # get some parameters
    num_classes = len(labels_id)
    num_label_samples = len(labels)
    num_unlabel_samples = Mat_Unlabel.shape[0]

    affinity_matrix_UL = affinity_matrix[:, 0:num_label_samples]
    affinity_matrix_UU = affinity_matrix[:, num_label_samples:num_label_samples + num_unlabel_samples]

    if comm_rank == 0:
        print("Have %d labeled images, %d unlabeled images and %d classes" % (
        num_label_samples, num_unlabel_samples, num_classes))

    # divide label_function_U and label_function_L to all processors
    class_offset = np.linspace(0, num_classes, comm_size + 1).astype('int')

    # initialize local label_function_U
    local_start_class = class_offset[comm_rank]
    local_num_classes = class_offset[comm_rank + 1] - local_start_class
    local_label_function_U = eye(num_unlabel_samples, local_num_classes, 0, np.float32, format='csr')

    # initialize local label_function_L
    local_label_function_L = lil_matrix((num_label_samples, local_num_classes), dtype=np.float32)
    for i in range(num_label_samples):
        class_off = int(labels[i]) - local_start_class
        if class_off >= 0 and class_off < local_num_classes:
            local_label_function_L[i, class_off] = 1.0
    local_label_function_L = local_label_function_L.tocsr()
    local_label_info = affinity_matrix_UL.dot(local_label_function_L)
    print("Processor %d/%d has to process %d classes..." % (comm_rank, comm_size, local_label_function_L.shape[1]))

    # start to propagation
    iter = 1
    changed = 100.0
    # evaluation(num_unlabel_samples, local_start_class, local_label_function_U, labels_id)
    while True:
        pre_label_function = local_label_function_U.copy()

        # propagation
        local_label_function_U = affinity_matrix_UU.dot(local_label_function_U) + local_label_info

        # check converge
        local_changed = abs(pre_label_function - local_label_function_U).sum()
        changed = comm.reduce(local_changed, root=0, op=MPI.SUM)
        status = 'RUN'
        test = False
        if comm_rank == 0:
            if iter % 1 == 0:
                norm_changed = changed / (num_unlabel_samples * num_classes)
                print("---> Iteration %d/%d, changed: %f" % (iter, max_iter, norm_changed))
            if iter >= max_iter or changed < tol:
                status = 'STOP'
                print("************** Iteration over! ****************")
            if iter % test_per_iter == 0:
                test = True
            iter += 1
        test = comm.bcast(test if comm_rank == 0 else None, root=0)
        status = comm.bcast(status if comm_rank == 0 else None, root=0)
        if status == 'STOP':
            break
        if test == True:
            unlabel_data_labels = evaluation(num_unlabel_samples, local_start_class, local_label_function_U, labels_id)
            if comm_rank == 0:
                file_name = "./label_propagation/pseudo_label_for_no_label" + "_iter" + str(iter) + ".txt"
                file = open(file_name, "w")
                for idx in range(len(unlabel_data_labels)):
                    line = unlabel_img_path[idx] + ':' + str(int(unlabel_data_labels[idx])) + '\n'
                    file.writelines(line)
                file.close()
    # unlabel_data_labels = evaluation(num_unlabel_samples, local_start_class, local_label_function_U, labels_id)
    return unlabel_data_labels


# main function
if __name__ == "__main__":
    Mat_Label = np.load(Mat_Label_file)
    Mat_Unlabel = np.load(Mat_Unlabel_file)
    labels = np.load(labels_file)
    unlabel_img_path = np.load(unlabel_img_path_file)

    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    # unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)

    # To avoid the inconsistency of different matrix size in threads, ignore the last few unlabeled data
    num_unlabel_samples = int(Mat_Unlabel.shape[0] / comm_size) * comm_size
    if comm_rank == 0:
        print('Have {} input unlabel data, but only leave {} of them.'.format(Mat_Unlabel.shape[0], num_unlabel_samples))
    Mat_Unlabel = Mat_Unlabel[:num_unlabel_samples, :] # delete the last few samples

    if comm_rank == 0:
        print("start label propagation")
    unlabel_data_labels = label_propagation_sparse(Mat_Label, Mat_Unlabel, labels, unlabel_img_path, kernel_type='knn', knn_num_neighbors=3,
                                           max_iter=500)

    # if comm_rank == 0:
    #     file = open("pseudo_label_for_no_label.txt", "w")
    #     for idx in range(len(unlabel_data_labels)):
    #         line = unlabel_img_path[idx] + ':' + str(int(unlabel_data_labels[idx])) + '\n'
    #         file.writelines(line)
    #     file.close()