from test_multiple import all_expnames
import pandas as pd
import os
import numpy as np
import pickle

# create pd dataframe if it doesn't exist
try:
    # import from pickle file ttc_data.pkl
    ttc_data = pd.read_pickle('ttc_data.pkl')
except:
    ttc_data = pd.DataFrame(columns=['expname', 'time_to_convergence', 'test_accuracy'])

def get_ttc(expname):
    if os.path.exists('save/{}/clients.pkl'.format(expname)):
        clients = pd.read_pickle('save/{}/clients.pkl'.format(expname))
    else:
        return None, None
    
    stops = []
    for client in clients:
        if expname.startswith('TOY'):
            stops.append(len(client.val_loss_list))
        else:
            stops.append(len(client.val_acc_list))
    return np.mean(stops), np.sum(stops)

def get_test_accuracy(expname):
    # if experiment is cifar10 label
    if 'CIFAR_label' in expname:
        with open(f'save/{expname}/CIFAR_A_within.pkl', 'rb') as f:
            A_within = pickle.load(f)
        with open(f'save/{expname}/CIFAR_V_within.pkl', 'rb') as f:
            V_within = pickle.load(f)
        # Concatenate A_within and V_within
        test_accs = np.concatenate((A_within, V_within))

    # if experiment is cifar10 5 cluster
    elif 'CIFAR_5_clusters' in expname:
        with open('save/{}/CIFAR_acc_matrix.pkl'.format(expname), 'rb') as f:
            CIFAR_acc_matrix = pickle.load(f)

        test_accs = []
        for k in range(5):
            test_accs.extend(CIFAR_acc_matrix[k,k,:].tolist())

    # if experiment is fashion mnist
    elif 'fashion_MNIST' in expname:
        with open('save/{}/fashion_MNIST_acc_matrix.pkl'.format(expname), 'rb') as f:
            fashion_MNIST_acc_matrix = pickle.load(f)

        test_accs = fashion_MNIST_acc_matrix

    # if experiment is toy dataset
    elif 'TOY' in expname:
        with open('save/{}/toy_test_losses.pkl'.format(expname), 'rb') as f:
            toy_test_losses = pickle.load(f)

        test_accs = []
        for k in range (len(toy_test_losses)):
            test_accs.extend(toy_test_losses[k].tolist())


    # if experiment is CIFAR100
    elif 'HUNDRED' in expname:
        with open('save/{}/CIFAR100_acc_matrix.pkl'.format(expname), 'rb') as f:
            CIFAR100_acc_matrix = pickle.load(f)

        test_accs = CIFAR100_acc_matrix

    # if experiment is double mnist
    elif 'DOUBLE' in expname:
        print('double mnist')

    return np.mean(test_accs)

# add new data
for expname_collection in all_expnames:
    for expname in expname_collection:
        if expname not in ttc_data['expname'].values:
            print('Collecting data for {}'.format(expname))
            avg_ttc, total_comm = get_ttc(expname)
            if avg_ttc is None:
                print('No data for {}'.format(expname))
                continue
            test_acc = get_test_accuracy(expname)
            new_row = pd.DataFrame([{
                'expname': expname,
                'time_to_convergence': avg_ttc,
                'total_communication': total_comm,
                'test_accuracy': test_acc
            }])

            # Concatenate the new row with the existing DataFrame
            ttc_data = pd.concat([ttc_data, new_row], ignore_index=True)
        else:
            print('Data for {} already exists'.format(expname))

# save to pickle
ttc_data.to_pickle('ttc_data.pkl')
