python get_results.py -d pclreid \
    -a vmgn_hgnn \
    -j 0 \
    --test-batch 1000 \
    --gpu-devices 0 \
    --model-weight ./log/resnet50-xent/vmgn_hgnn1_submit.tar \
    --save-json ./vmgn_hgnn1_no_global_branch_submit.json