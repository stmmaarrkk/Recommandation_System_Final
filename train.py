import os
import numpy as np
from LinearReg import LinearReg
from LinearRegNorm import LinearRegNorm

def main():
    input_file = os.path.join("./dataset", 'nexus-6/SDD6_8-12.npz')
    data = np.load(input_file)
    # Data come as NxTx2 numpy nd-arrays where N is the number of trajectories,
    # T is their duration.
    dataset_obsv, dataset_pred, dataset_t, the_batches = \
        data['obsvs'], data['preds'], data['times'], data['batches']

    # 4/5 of the batches to be used for training phase, 1/5 for testing
    # 7/8 of the batches in training phase to be used for training, 1/8 for validation
    train_val_size = max(1, (len(the_batches) * 4) // 5)
    train_size = max(1, (train_val_size * 7) // 8)
    val_size = train_val_size - train_size
    test_size = len(the_batches) - train_val_size


    n_past = dataset_obsv.shape[1]  # Size of the observed sub-paths
    n_next = dataset_pred.shape[1]  # Size of the sub-paths to predict

    print("LinearReg:")
    clf = LinearReg()
    clf.fit(dataset_obsv[:train_val_size], dataset_pred[:train_val_size])
    clf.score(dataset_obsv[train_val_size:], dataset_pred[train_val_size:])

    print("LinearRegNorm:")
    clfNorm = LinearRegNorm()
    clfNorm.fit(dataset_obsv[:train_val_size], dataset_pred[:train_val_size])
    clfNorm.score(dataset_obsv[train_val_size:], dataset_pred[train_val_size:])

    n_total_samples = the_batches[-1][1]
    n_train_samples = the_batches[train_size - 1][1]  # Number of training samples

    print('Data path:', input_file)
    print(' # Total samples:', n_total_samples, ' # Training samples:', n_train_samples)

if __name__ == "__main__":
    main()