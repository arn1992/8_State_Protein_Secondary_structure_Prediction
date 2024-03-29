import numpy as np
import gzip
import h5py


def load_cul6133_filted():
    '''
    TRAIN data Cullpdb+profile_6133_filtered
    Test data  CB513 CASP10 CASP11
    '''
    print("Loading train data (Cullpdb_filted)...")
    data = np.load('data/cullpdb+profile_6133_filtered.npy.gz')
    data = np.reshape(data, (-1, 700, 57))
    # print data.shape
    datahot = data[:, :, 0:21]  # sequence feature
    # print 'sequence feature',dataonehot[1,:3,:]
    datapssm = data[:, :, 35:56]  # profile feature
    # print 'profile feature',datapssm[1,:3,:]
    labels = data[:, :, 22:30]  # secondary struture label , 8-d
    np.random.seed(2018)
    # shuffle data
    num_seqs, seqlen, feature_dim = np.shape(data)
    num_classes = labels.shape[2]
    seq_index = np.arange(0, num_seqs)  #
    np.random.shuffle(seq_index)

    # train data
    trainhot = datahot[seq_index[:5278]]  # 21
    trainlabel = labels[seq_index[:5278]]  # 8
    trainpssm = datapssm[seq_index[:5278]]  # 21

    # val data
    vallabel = labels[seq_index[5278:5534]]  # 8
    valpssm = datapssm[seq_index[5278:5534]]  # 21
    valhot = datahot[seq_index[5278:5534]]  # 21

    train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
    for i in range(trainhot.shape[0]):
        for j in range(trainhot.shape[1]):
            if np.sum(trainhot[i, j, :]) != 0:
                train_hot[i, j] = np.argmax(trainhot[i, j, :])

    val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
    for i in range(valhot.shape[0]):
        for j in range(valhot.shape[1]):
            if np.sum(valhot[i, j, :]) != 0:
                val_hot[i, j] = np.argmax(valhot[i, j, :])

    return train_hot, trainpssm, trainlabel, val_hot, valpssm, vallabel


def load_cb513():
    print("Loading Test data (CB513)...")
    CB513 = np.load('data/cb513+profile_split1.npy.gz')
    CB513 = np.reshape(CB513, (-1, 700, 57))
    # print CB513.shape
    datahot = CB513[:, :, 0:21]  # sequence feature
    datapssm = CB513[:, :, 35:56]  # profile feature

    labels = CB513[:, :, 22:30]  # secondary struture label
    testhot = datahot
    testlabel = labels
    testpssm = datapssm

    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i, j, :]) != 0:
                test_hot[i, j] = np.argmax(testhot[i, j, :])

    return test_hot, testpssm, testlabel


def load_casp10():
    print("Loading Test data (CASP10)...")
    casp10 = h5py.File("data/casp10.h5")
    # print casp10.shape
    datahot = casp10['features'][:, :, 0:21]  # sequence feature
    datapssm = casp10['features'][:, :, 21:42]  # profile feature
    labels = casp10['labels'][:, :, 0:8]  # secondary struture label

    testhot = datahot
    testlabel = labels
    testpssm = datapssm

    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i, j, :]) != 0:
                test_hot[i, j] = np.argmax(testhot[i, j, :])

    return test_hot, testpssm, testlabel


def load_casp11():
    print("Loading Test data (CASP11)...")
    casp11 = h5py.File("data/casp11.h5")
    # print casp11.shape
    datahot = casp11['features'][:, :, 0:21]  # sequence feature
    datapssm = casp11['features'][:, :, 21:42]  # profile feature
    labels = casp11['labels'][:, :, 0:8]  # secondary struture label
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i, j, :]) != 0:
                test_hot[i, j] = np.argmax(testhot[i, j, :])
    return test_hot, testpssm, testlabel