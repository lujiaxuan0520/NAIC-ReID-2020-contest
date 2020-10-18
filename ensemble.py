####################################################################
# ensemble distmaps of different models and get the ensemble result.
# Author: Jiaxuan Lu
####################################################################

import numpy as np
import json

base_dir = './results/'

save_json = "ensemble_6.json"
# model_name = ['vmgn_hgnn4_best_rerank_submit',
#               'vmgn_hgnn13_best_rerank_submit',
#               'efn0_hgnn1_best_rerank_submit',
#               'efn0_hgnn1_iter_80_rerank_submit',
#               'efn3_hgnn2_best_rerank_submit',
#               'efn5_hgnn3_best_rerank_submit',
#               'efn0_best_rerank_submit',
#               'vmgn_hgnn14_best_rerank_submit',
#               'efn0_hgnn4_best_rerank_submit',
#               'vmgn_hgnn15_best_rerank_submit',
#               'vmgn_hgnn16_best_rerank_submit',
#               'vmgn_hgnn16_best_rerank2_submit']
# model_weight = [0.52, 0.54, 0.48, 0.47, 0.48, 0.47, 0.45, 0.51, 0.47, 0.48, 0.51, 0.52] # the weight of different models
# model_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ] # the weight of different models
model_name = ['vmgn_hgnn4_best_rerank_submit','vmgn_hgnn13_best_rerank_submit']
model_weight = [1.0, 1.0]


class Ensemble():
    def __init__(self, base_dir, save_json, model_name, model_weight):
        self.base_dir = base_dir
        self.save_json = save_json
        self.model_name = model_name
        self.model_weight = model_weight
        self.q_index_map = dict()  # map, ex: '00000018.png':0
        self.g_index_map = dict()  # map, ex: '00115338.png':0
        self.final_distmap = np.zeros([])

    def ensemble(self):
        for model_idx, model in enumerate(self.model_name):
            distmap_file = model + ".npy"
            query_file = model + "_q_img_paths.npy"
            gallery_file = model + "_g_img_paths.npy"

            distmap = np.load(self.base_dir + distmap_file)
            query_name = np.load(self.base_dir + query_file)
            gallery_name = np.load(self.base_dir + gallery_file)

            if model_idx == 0:
                first_query_name = query_name
                first_gallery_name = gallery_name
                self.final_distmap = self.model_weight[0] * distmap  # initialize the final_distmap
                for idx, line in enumerate(distmap):
                    query = query_name[idx]  # ex: '00000018.png'
                    self.q_index_map[query] = idx
                    for jdx, dist in enumerate(line):
                        gallery = gallery_name[jdx]  # ex: '00115338.png'
                        self.g_index_map[gallery] = jdx
            else:  # add the dist to final_distmap
                for idx, line in enumerate(distmap):
                    new_i_index = self.q_index_map[query_name[idx]]
                    for jdx, dist in enumerate(line):
                        new_j_index = self.g_index_map[gallery_name[jdx]]
                        self.final_distmap[new_i_index][new_j_index] += self.model_weight[model_idx] * dist

            print("{} has been added.".format(model))

        res_dict = dict()
        for query_idx, line in enumerate(self.final_distmap):
            query_name = first_query_name[query_idx]
            gallery_top_200_idx = np.argsort(line)[:200]  # the index of the top 200 similiar images in gallery
            gallery_top_200_name = [first_gallery_name[item] for item in gallery_top_200_idx]
            res_dict[query_name] = gallery_top_200_name

        # save the json results
        json_str = json.dumps(res_dict)
        with open(self.save_json, 'w') as json_file:
            json_file.write(json_str)

        print("Done.")


E = Ensemble(base_dir, save_json, model_name, model_weight)
E.ensemble()
