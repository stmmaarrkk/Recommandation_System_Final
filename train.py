import os
import numpy as np
from LinearReg import LinearReg
from LinearRegNorm import LinearRegNorm
from LinearRegRecur import LinearRegRecur
import helper

def main():
    input_file = os.path.join("./dataset", 'nexus-6/SDD6_8-12.npz')
    #input_file = os.path.join("./dataset", 'SDD9-8-12.npz')
    data = np.load(input_file)
    # Data come as NxTx2 numpy nd-arrays where N is the number of trajectories,
    # T is their duration.
    dataset_obsv, dataset_pred, dataset_t, the_batches = \
        data['obsvs'], data['preds'], data['times'], data['batches']

    dataset_obsv, dataset_pred = helper.normalization(dataset_obsv, dataset_pred)

    # 4/5 of the batches to be used for training phase, 1/5 for testing
    # 7/8 of the batches in training phase to be used for training, 1/8 for validation
    train_size = data['obsvs'].shape[0] * 4 // 5

    print("LinearReg:")
    clf = LinearReg()
    clf.fit(dataset_obsv[:train_size].copy(), dataset_pred[:train_size].copy())
    clf.score(dataset_obsv[train_size:].copy(), dataset_pred[train_size:].copy())

    print("LinearRegNorm:")
    clfNorm = LinearRegNorm()
    clfNorm.fit(dataset_obsv[:train_size].copy(), dataset_pred[:train_size].copy())
    clfNorm.score(dataset_obsv[train_size:].copy(), dataset_pred[train_size:].copy())

    print("LinearRegRecur:")
    clfNormRecur = LinearRegRecur(5, 13)
    clfNormRecur.fit(dataset_obsv[:train_size].copy(), dataset_pred[:train_size].copy())
    clfNormRecur.score(dataset_obsv[train_size:].copy(), dataset_pred[train_size:].copy())

if __name__ == "__main__":
    main()