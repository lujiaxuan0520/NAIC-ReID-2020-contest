python train_imgreid_xent_htri.py -d pclreid \
    -a resnet50 \
    --optim adam \
    --lr 0.003 \
    --max-epoch 200 \
    --train-batch 32 \
    --test-batch 100 \
    --save-dir ./log/resnet50-xent \
    --gpu-devices 5 \
    --eval-step 1
