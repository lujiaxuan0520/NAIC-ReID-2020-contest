python train_imgreid_xent_htri.py -d pclreid \
    -a vmgn_hgnn \
    --optim adam \
    --lr 0.0002 \
    --max-epoch 80 \
    --train-batch 64 \
    --test-batch 100 \
    --soft-margin \
    --label-smooth \
    --warmup \
    --save-dir ./log/resnet50-xent \
    --gpu-devices 0 \
    --eval-step 1
