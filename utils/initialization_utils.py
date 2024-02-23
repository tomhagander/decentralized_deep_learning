import numpy as np

def sample_cifargroups(dataset, num_users, n_data_train, n_data_val):

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
        if(i<int(num_users*0.4)): #vehicles
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