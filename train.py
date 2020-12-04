import os
import numpy as np
from LinearReg import LinearReg
from LinearRegNorm import LinearRegNorm
from LinearRegRecur import LinearRegRecur
from utils.Preprocessing import Scale
from Evaluator import Evaluator

def main():
    input_file = os.path.join("./dataset", 'nexus-6/SDD6_8-12.npz')
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

    evaluator = Evaluator(scale.sx, scale.sy);

    # 4/5 of the batches to be used for training phase, 1/5 for testing
    # 7/8 of the batches in training phase to be used for training, 1/8 for validation
    train_size = data['obsvs'].shape[0] * 4 // 5

    print("LinearReg:")
    clf = LinearReg(evaluator)
    clf.fit(dataset_obsv[:train_size].copy(), dataset_pred[:train_size].copy())
    clf.score(dataset_obsv[train_size:].copy(), dataset_pred[train_size:].copy())

    print("LinearRegRecur:")
    clfNormRecur = LinearRegRecur(evaluator, 5, 13)
    clfNormRecur.fit(dataset_obsv[:train_size].copy(), dataset_pred[:train_size].copy())
    clfNormRecur.score(dataset_obsv[train_size:].copy(), dataset_pred[train_size:].copy())

if __name__ == "__main__":
    main()