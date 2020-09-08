####################################################################################
# initialize the datasets, including train, gallery and query
# Author: Jiaxuan Lu
# make the folder as:
#   images: directory, the training images
#   query: directory, the query images
#   gallery: directory, the gallery images
#   label.txt: file, the label of training images
# output:
#   if isFinal: train_list.txt, query_list.txt, gallery_list.txt, no_label_list.txt
#   else: train_list.txt, query_list_final.txt, gallery_list_final.txt, no_label_list.txt
####################################################################################
import os
import os.path as osp
import random

isFinal = False # true: generate gallery_list_final.txt and query_list_final.txt
all_as_train = False # true: let all the images in label.txt be the training images

dataset_root = "./PCL_ReID"
label_file = osp.join(dataset_root, 'label.txt')
train_file = osp.join(dataset_root, 'train_list.txt')
query_file = osp.join(dataset_root, 'query_list_final.txt') if isFinal else osp.join(dataset_root, 'query_list.txt')
gallery_file = osp.join(dataset_root, 'gallery_list_final.txt') if isFinal else osp.join(dataset_root, 'gallery_list.txt')
no_label_file = osp.join(dataset_root, 'no_label_list.txt')

# read the label.txt to pid_dict
with open(label_file, 'r') as txt:
    lines = txt.readlines()
pid_dict = dict() # key: img_name, value: person_id
for img_idx, img_info in enumerate(lines):
    img_path, pid = img_info.split(':')
    pid = int(pid)
    pid_dict[img_path] = pid

# write to gallery_list.txt or gallery_list_final.txt
file = open(gallery_file,"w")
if isFinal:
    pre = osp.join(dataset_root, "gallery")
    file_list = os.listdir(pre)
    for key in file_list:
        img_path = osp.join(pre, key)
        line = img_path + '\n'
        file.writelines(line)
else:
    pre = osp.join(dataset_root, "images")
    for key, value in pid_dict.items():
        img_path = osp.join(pre, key)
        line = img_path + ':' + str(value) + '\n'
        file.writelines(line)
file.close()

# write to query_list.txt or query_list_final.txt
file = open(query_file,"w")
query_set = set() # save all the image name in the query dataset
if isFinal:
    pre = osp.join(dataset_root, "query")
    file_list = os.listdir(pre)
    for key in file_list:
        img_path = osp.join(pre, key)
        line = img_path + '\n'
        file.writelines(line)
else:
    query_num = int(0.1 * len(pid_dict)) # choose 10% images as query dataset
    pre = osp.join(dataset_root, "images")
    keys = random.sample(pid_dict.keys(), query_num)  # randomly choose keys
    for key in keys:
        query_set.add(key)
        img_path = osp.join(pre, key)
        line = img_path + ':' + str(pid_dict[key]) + '\n'
        file.writelines(line)
file.close()

# write to train_list.txt
file = open(train_file,"w")
pre = osp.join(dataset_root,"images")
for key, value in pid_dict.items():
    if all_as_train: # all of the images as training set
        img_path = osp.join(pre, key)
        line = img_path + ':' + str(value) + '\n'
        file.writelines(line)
    else: # choose the images not in query as training set
        if key in query_set: # do not add to the training set
            continue
        else:
            img_path = osp.join(pre, key)
            line = img_path + ':' + str(value) + '\n'
            file.writelines(line)
file.close()

# write to no_label.txt
file = open(no_label_file,"w")
pre = osp.join(dataset_root, "images")
file_list = os.listdir(pre)
for key in file_list:
    if key not in pid_dict.keys():
        img_path = osp.join(pre, key)
        line = img_path + '\n'
        file.writelines(line)
file.close()