import os
import numpy as np
from LinearReg import LinearReg
from LinearRegNorm import LinearRegNorm
from LinearRegRecur import LinearRegRecur
from utils.Preprocessing import Scale
from Evaluator import Evaluator

def main():
    for x in range (1, 3):
        input_file = os.path.join("./dataset", "nexus-" + str(x) + "/SDD" + str(x) + "_type-8-12.npz")
        print("================="+str(x)+"====================")
        #input_file = os.path.join("./dataset", 'SDD9-8-12.npz')
        data = np.load(input_file)
        # Data come as NxTx2 numpy nd-arrays where N is the number of trajectories,
        # T is their duration.
        dataset_obsv, dataset_pred, dataset_t, the_batches = \
            data['obsvs'], data['preds'], data['times'], data['batches']

        # ================ Normalization ================
        scale = Scale()
        scale.max_x = max(np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
        scale.min_x = min(np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
        scale.max_y = max(np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
        scale.min_y = min(np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
        scale.calc_scale(keep_ratio=True)
        dataset_obsv = scale.normalize(dataset_obsv)
        dataset_pred = scale.normalize(dataset_pred)

        evaluator = Evaluator(scale.sx, scale.sy)

        # 4/5 of the batches to be used for training phase, 1/5 for testing
        # 7/8 of the batches in training phase to be used for training, 1/8 for validation
        train_size = data['obsvs'].shape[0] * 4 // 5
        print(data['obsvs'].shape[0])
        print("LinearReg:")
        clf = LinearReg(evaluator)
        clf.fit(dataset_obsv[:train_size,:,:2].copy(), dataset_pred[:train_size,:,:2].copy())
        clf.score(dataset_obsv[train_size:,:,:2].copy(), dataset_pred[train_size:,:,:2].copy())

        print("LinearRegRecur:")
        clfNormRecur = LinearRegRecur(evaluator, epoch=5, batchSize=11)
        clfNormRecur.fit(dataset_obsv[:train_size,:,:2].copy(), dataset_pred[:train_size,:,:2].copy())
        clfNormRecur.score(dataset_obsv[train_size:,:,:2].copy(), dataset_pred[train_size:,:,:2].copy())

if __name__ == "__main__":
    main()