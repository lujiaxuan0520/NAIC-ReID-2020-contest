python get_results.py -d pclreid \
    -a vmgn_hgnn \
    -j 0 \
    --test-batch 1000 \
    --gpu-devices 5 \
    --global-branch \
    --dist-metric cosine \
    --re-rank \
    --model-weight ./log/resnet50-xent/checkpoint_ep80.pth.tar \
    --save-json ./vmgn_hgnn3_best_submit.json