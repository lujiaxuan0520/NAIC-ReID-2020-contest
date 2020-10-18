################################################################################################
# semi-supervised learning: use label propagation to make pseudo labels for no label data
# This is the parallel version of label propagation, required openmpi and Cython
# Author: Jiaxuan Lu
# run label_propagation_mpi_get_feature.py first, and then
#   run "mpirun -np 5 label_propagation_mpi.py", finally run label_propagation_postprocessing.py
################################################################################################
import os, sys
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


extended_data = True
input_propagation_name = "./label_propagation/pseudo_label_for_no_label_iter491.txt"
input_train_name = "./PCL_ReID/train_extended_list.txt" if extended_data else "./PCL_ReID/train_list.txt"
output_file_name = "./PCL_ReID/train_extended_pseudo_list.txt" if extended_data else "./PCL_ReID/train_pseudo_list.txt"
current_max_id = -1

# write the file as same as train_list.txt
file = open(output_file_name,"w")
with open(input_train_name, 'r') as txt:
    lines = txt.readlines()
for img_idx, img_info in enumerate(lines):
    img_path, pid = img_info.split(':')
    pid = int(pid)
    if pid > current_max_id:
        current_max_id = pid
    file.writelines(img_info)

label_trans_map = dict() # transform the label (old_pid: new_pid)
# add the pseudo label as new label
with open(input_propagation_name, 'r') as txt:
    lines = txt.readlines()
for img_idx, img_info in enumerate(lines):
    img_path, old_pid = img_info.split(':')
    old_pid = int(old_pid)
    if old_pid in label_trans_map:
        pid = label_trans_map[old_pid]
    else:
        current_max_id += 1
        pid = current_max_id
        label_trans_map[old_pid] = pid
    line = img_path + ':' + str(pid) + '\n'
    file.writelines(line)

file.close()

print('Done, the current max pid is {}'.format(current_max_id))
