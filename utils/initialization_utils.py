import numpy as np
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import random
import os


def sample_cifargroups(dataset, num_users, n_data_train, n_data_val, ratio):
    # set random seed

    group1 = np.array([0,1,8,9]) #vehicles
    group2 = np.array([2,3,4,5,6,7]) #animals
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    idxs1, idxs2 = np.array([]), np.array([])
    idxs1 = idxs1.astype(int)
    idxs2 = idxs1.astype(int)
    for x in group1:
        idxs1 = np.append(idxs1, idxs[x == labels[idxs]])
    
    for x in group2:
        idxs2 = np.append(idxs2, idxs[x == labels[idxs]])
    
    for i in range(num_users):
        if(i<int(num_users*ratio)): #vehicles
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
            
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
        else: #animals
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
            
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))

    return dict_users, dict_users_val

def sample_cifargroups_100(dataset, num_users, n_data_train, n_data_val, ratio):

    cluster1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 75, 76, 77, 78, 79, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84])
    cluster2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 85, 86, 87, 88, 89, 50, 51, 52, 53, 54])
    cluster3 = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 45, 46, 47, 48, 49])

    print('Length of cluster1:', len(cluster1))
    print('Length of cluster2:', len(cluster2))
    print('Length of cluster3:', len(cluster3))
    

    # Initialize dictionaries
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    # Sort indices by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    # Initialize index arrays for each cluster
    idxs1, idxs2, idxs3 = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    # Populate index arrays based on cluster definitions
    for x in cluster1:
        idxs1 = np.append(idxs1, idxs[x == labels[idxs]])
    for x in cluster2:
        idxs2 = np.append(idxs2, idxs[x == labels[idxs]])
    for x in cluster3:
        idxs3 = np.append(idxs3, idxs[x == labels[idxs]])

    # Assign data indices to users
    for i in range(num_users):
        if i < int(num_users * 0.5):
            cluster_choice = 1
        elif i < int(num_users * 0.75):
            cluster_choice = 2
        else:
            cluster_choice = 3

        if cluster_choice == 1:
            selected_idxs = np.random.choice(idxs1, n_data_train + n_data_val, replace=False)
        elif cluster_choice == 2:
            selected_idxs = np.random.choice(idxs2, n_data_train + n_data_val, replace=False)
        else:
            selected_idxs = np.random.choice(idxs3, n_data_train + n_data_val, replace=False)
        
        dict_users[i] = selected_idxs[:n_data_train]
        dict_users_val[i] = selected_idxs[n_data_train:n_data_train + n_data_val]

        # Update remaining indices to prevent reselection
        if cluster_choice == 1:
            idxs1 = np.setdiff1d(idxs1, selected_idxs)
        elif cluster_choice == 2:
            idxs2 = np.setdiff1d(idxs2, selected_idxs)
        else:
            idxs3 = np.setdiff1d(idxs3, selected_idxs)

    return dict_users, dict_users_val


def sample_cifargroups_5clusters(dataset, num_users, n_data_train, n_data_val, ratio):
    # set random seed

    group1 = np.array([0,1])
    group2 = np.array([2,3])
    group3 = np.array([4,5])
    group4 = np.array([6,7])
    group5 = np.array([8,9])

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    idxs1, idxs2, idxs3, idxs4, idxs5 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    idxs1 = idxs1.astype(int)
    idxs2 = idxs1.astype(int)
    idxs3 = idxs1.astype(int)
    idxs4 = idxs1.astype(int)
    idxs5 = idxs1.astype(int)
    for x in group1:
        idxs1 = np.append(idxs1, idxs[x == labels[idxs]])
    
    for x in group2:
        idxs2 = np.append(idxs2, idxs[x == labels[idxs]])

    for x in group3:
        idxs3 = np.append(idxs3, idxs[x == labels[idxs]])

    for x in group4:
        idxs4 = np.append(idxs4, idxs[x == labels[idxs]])

    for x in group5:
        idxs5 = np.append(idxs5, idxs[x == labels[idxs]])

    for i in range(num_users):
        if(i<int(num_users*ratio)):
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))

            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
        elif(i<int(num_users*2*ratio)):
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))

            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
        elif(i<int(num_users*3*ratio)):
            sub_data_idxs3 = np.random.choice(idxs3, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs3)))
            idxs3 = np.array(list(set(idxs3) - set(sub_data_idxs3)))

            sub_data_idxs3 = np.random.choice(idxs3, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs3)))
            idxs3 = np.array(list(set(idxs3) - set(sub_data_idxs3)))
        elif(i<int(num_users*4*ratio)):
            sub_data_idxs4 = np.random.choice(idxs4, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs4)))
            idxs4 = np.array(list(set(idxs4) - set(sub_data_idxs4)))

            sub_data_idxs4 = np.random.choice(idxs4, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs4)))
            idxs4 = np.array(list(set(idxs4) - set(sub_data_idxs4)))
        else:
            sub_data_idxs5 = np.random.choice(idxs5, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs5)))
            idxs5 = np.array(list(set(idxs5) - set(sub_data_idxs5)))

            sub_data_idxs5 = np.random.choice(idxs5, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs5)))
            idxs5 = np.array(list(set(idxs5) - set(sub_data_idxs5)))
        
    return dict_users, dict_users_val

def sample_labels_iid(dataset, num_users, n_data_train, n_data_val):
    """
    Sample I.I.D. (labels) client data from MNIST/CIFAR10/FASHION-MNIST datasets
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_val = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, int(n_data_train), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        dict_users_val[i] = set(np.random.choice(all_idxs, int(n_data_val), replace=False))
        all_idxs = list(set(all_idxs) - dict_users_val[i])
        
    return dict_users, dict_users_val

def uniform_split(dataset, num_users, n_data_train, n_data_val):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    for i in range(num_users):
        sub_data_idxs = np.random.choice(idxs, int(n_data_train+n_data_val), replace=False)
        dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs[:n_data_train])))
        dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs[n_data_train:])))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs)))
        
    return dict_users, dict_users_val


def load_pickle_files(path, filenames):
    loaded_files = []
    for filename in filenames:
        with open(path + filename, 'rb') as f:
            loaded_files.append(pickle.load(f))
    return loaded_files

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def split_dataset(dataset, n_clients):  
    dataset_length = len(dataset)
    splits = np.array_split(np.arange(dataset_length), n_clients)
    dataset_sizes = [len(split) for split in splits]
    datasets = torch.utils.data.random_split(dataset, lengths=dataset_sizes)
    print(len(datasets[-1]))
    return datasets


def load_pacs(path, BATCH_SIZE, nbr_clients_per_group, augment=False):
    # means and standard deviations ImageNet because the network is pretrained
    means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # Define transforms to apply to each image
    transf = transforms.Compose([ 
                                transforms.Resize((224,224)),  # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
                                transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
                                ])
    
    if(augment):
        transf_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transf_train = transf
    

    #load pickle files, create datasets and apply transforms
    filenames = ['photo_train.pkl', 'photo_val.pkl', 'photo_test.pkl', 'art_train.pkl', 'art_val.pkl', 
                    'art_test.pkl', 'cartoon_train.pkl', 'cartoon_val.pkl', 'cartoon_test.pkl', 'sketch_train.pkl', 'sketch_val.pkl', 'sketch_test.pkl']
    photo_train, photo_val, photo_test, art_train, art_val, art_test, cartoon_train, cartoon_val, cartoon_test, sketch_train, sketch_val, sketch_test = load_pickle_files(path, filenames)
    train_datasets = [photo_train, art_train, cartoon_train, sketch_train]
    val_datasets = [photo_val, art_val, cartoon_val, sketch_val]
    test_datasets = [photo_test, art_test, cartoon_test, sketch_test]
    for dataset in train_datasets:
        dataset.transform = transf_train
    for dataset in val_datasets:
        dataset.transform = transf
    for dataset in test_datasets:
        dataset.transform = transf

    # split datasets
    client_train_datasets = []
    for trainset in train_datasets:
        splits = split_dataset(trainset, nbr_clients_per_group)
        for s in splits:
            client_train_datasets.append(s)

    return client_train_datasets, val_datasets, test_datasets