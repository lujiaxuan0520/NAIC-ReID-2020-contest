python get_results.py -d pclreid \
    -a vmgn_hgnn \
    -j 0 \
    --test-batch 1000 \
    --gpu-devices 5 \
    --model-weight ./log/resnet50-xent/checkpoint_ep97.pth.tar \
    --save-json ./vmgn_hgnn1_submit.json