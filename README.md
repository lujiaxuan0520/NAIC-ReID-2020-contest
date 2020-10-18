# Introduction
This is the code for ReID-2020 contest in the first stage, ranking 89 in Leardership B.

+ Hypergraph learning for feature propagation.
+ Tricks including soft margin, label smooth and warmup.
+ Ensemble with different models including ResNet 50 and ResNet 101.

# How to use
1. Put the datasets into ./PCL_ReID/images, ./PCL_ReID/query and ./PCL_ReID/gallery.
2. Run init_dataset.py to get the query_list.txt, query_list_final.txt, gallery_list.txt, gallery_list_final.txt and 
no_label_list.txt.
3. Run `bash run.sh`, or just run train_imgreid_xent_htri.py with parameters.
4. Train two models, one in ResNet 50 and the other in ResNet 101 with extended data (ReID-2019). Or you can download 
the models from [BaiduYun](https://pan.baidu.com/s/10yMrLbxn2Djok-5UDRcpvw) with code ihwh.
5. Run `bash get_results.sh` to get the json results of two models.
6. Run ensemble.py to get the ensemble results.



