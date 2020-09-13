python get_results.py -d pclreid \
    -a vmgn_hgnn \
    -j 0 \
    --test-batch 1000 \
    --gpu-devices 7 \
    --global-branch \
    --dist-metric cosine \
    --re-rank \
    --model-weight ./log/resnet50-xent/checkpoint_ep75.pth.tar \
    --save-json ./vmgn_hgnn6_best_submit.json