python get_results.py -d pclreid \
    -a vmgn_hgnn \
    -j 0 \
    --test-batch 1000 \
    --gpu-devices 6 \
    --model-weight ./log/resnet50-xent/checkpoint_ep39.pth.tar \
    --save-json ./tmp_submit.json