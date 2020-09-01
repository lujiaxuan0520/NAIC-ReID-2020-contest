python get_results.py -d pclreid \
    -a resnet50 \
    -j 0 \
    --test-batch 100 \
    --gpu-devices 5 \
    --model-weight ./log/resnet50-xent/baseline_model.pth.tar \
    --save-json ./baseline_submit.json