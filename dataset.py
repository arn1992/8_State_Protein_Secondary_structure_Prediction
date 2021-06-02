
import numpy as np

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

if __name__ == '__main__':
    dataset = get_dataset()

    D_train, D_test, D_val = split_with_shuffle(dataset, 100)

    X_train, Y_train = get_data_labels(D_train)
    X_test, Y_test = get_data_labels(D_test)
    X_val, Y_val = get_data_labels(D_val)

    print("Dataset Loaded")