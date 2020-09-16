python get_results.py -d pclreid \
    -a vmgn_hgnn \
    -j 0 \
    --test-batch 500 \
    --gpu-devices 0 \
    --global-branch \
    --dist-metric cosine \
    --re-rank \
    --model-weight ./log/resnet50-xent/vmgn_hgnn10/checkpoint_ep60.pth.tar \
    --save-json ./vmgn_hgnn10_iter_60_rerank_submit.json \
