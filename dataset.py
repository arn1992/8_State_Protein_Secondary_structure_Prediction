
import numpy as np
import  h5py

dataset_path = "data/cullpdb+profile_6133.npy.gz"
#dataset_path = "data/cullpdb+profile_6133_filtered.npy.gz"

cb513_path = "data/cb513+profile_split1.npy.gz"

sequence_len = 700
total_features = 57
amino_acid_residues = 21
num_classes = 8


def get_dataset(path=dataset_path):
    ds = np.load(path)
    ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))
    ret = np.zeros((ds.shape[0], ds.shape[1], amino_acid_residues + num_classes))
    ret[:, :, 0:amino_acid_residues] = ds[:, :, 35:56]
    ret[:, :, amino_acid_residues:] = ds[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + num_classes]
    return ret


def get_data_labels(D):
    X = D[:, :, 0:amino_acid_residues]
    Y = D[:, :, amino_acid_residues:amino_acid_residues + num_classes]
    return X, Y

'''def split_like_paper(Dataset):
    # Dataset subdivision following dataset readme and paper
    Train = Dataset[0:4300]
    Test = Dataset[4300:5017]
    Validation = Dataset[5017:5534]
    return Train, Test, Validation'''

def split_like_paper(Dataset):
    # Dataset subdivision following dataset readme and paper
    Train = Dataset[0:5600]
    Test = Dataset[5605:5877]
    Validation = Dataset[5877:6133]
    return Train, Test, Validation


def split_with_shuffle(Dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    train_split = int(Dataset.shape[0]*0.8)
    test_val_split = int(Dataset.shape[0]*0.1)
    Train = Dataset[0:5600, :, :]
    Test = Dataset[5605:5877, :, :]
    Validation = Dataset[5877:6133, :, :]
    return Train, Test, Validation

'''def split_with_shuffle(Dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    train_split = int(Dataset.shape[0]*0.8)
    test_val_split = int(Dataset.shape[0]*0.1)
    Train = Dataset[0:4300, :, :]
    Test = Dataset[4300:4917, :, :]
    Validation = Dataset[4917:5534, :, :]
    return Train, Test, Validation'''


def get_cb513():
    CB = get_dataset(cb513_path)
    X, Y = get_data_labels(CB)
    return X, Y

def get_casp10():
    print("casp10")
    #CB = get_dataset(casp10_path)
    casp10 = h5py.File("data/casp10.h5")

    datahot = casp10['features'][:, :, 0:21]  # sequence feature
    datapssm = casp10['features'][:, :, 21:42]  # profile feature
    labels = casp10['labels'][:, :, 0:8]  # secondary struture label
    #datahot=
    print(datahot)

    testhot = datahot
    testlabel = labels
    testpssm = datapssm

    #print(testlabel.shape,testhot.shape)
    #X, Y = get_data_labels(CB)
    #print(X.shape, Y.shape)
    return testhot,testlabel
def get_casp11():
    print("Loading Test data (CASP11)...")
    casp11 = h5py.File("data/casp11.h5")
    # print casp11.shape
    datahot = casp11['features'][:, :, 0:21]  # sequence feature
    print('datahot',datahot)
    datapssm = casp11['features'][:, :, 21:42]  # profile feature
    labels = casp11['labels'][:, :, 0:8]  # secondary struture label
    print('label',labels)
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    return testhot, testlabel
def get_casp10_1():
    casp10 = h5py.File("data/casp10.h5")
    casp10_feature = casp10['features'][:, :, 0:42].astype("float32")

    casp10_labels_1 = casp10['labels'][:, :, 0:8]
    num_seqs, seqlen, feature_dim = np.shape(casp10_feature)
    num_classes = 8
    vals = np.arange(0, 8)
    # secondary structure label
    labels_new_1 = np.zeros((num_seqs, seqlen))
    for i in range(num_seqs):
        labels_new_1[i, :] = np.dot(casp10_labels_1[i, :, :], vals)
    casp10_labels_1 = labels_new_1.astype('int32')

    casp10_mask = 1 - casp10['features'][:, :, -1].astype('uint8')
    print(casp10_labels_1)
    print(casp10_mask)

    print("casp10 data shape is", casp10_feature.shape, ", labels shape is", casp10_labels_1.shape, \
          ", mask shape is", casp10_mask.shape, "\n")

    #print("load all data takes time %fs" % (time.time() - time_start))

if __name__ == '__main__':
    dataset = get_dataset()

    D_train, D_test, D_val = split_with_shuffle(dataset, 100)

    X_train, Y_train = get_data_labels(D_train)
    X_test, Y_test = get_data_labels(D_test)
    X_val, Y_val = get_data_labels(D_val)

    print("Dataset Loaded")