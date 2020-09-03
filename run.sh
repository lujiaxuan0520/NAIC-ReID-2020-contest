python train_imgreid_xent_htri.py -d pclreid \
    -a vmgn_hgnn \
    --optim adam \
    --lr 0.0003 \
    --max-epoch 300 \
    --train-batch 64 \
    --test-batch 100 \
    --save-dir ./log/resnet50-xent \
    --gpu-devices 5 \
    --eval-step 1
