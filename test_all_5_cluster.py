

# go through each folder in '/save' and save the foldernames to a list if they start with 'CIFAR_5_clusters'

import os
import pickle
import subprocess
import pandas
import numpy as np

# get the list of folders in '/save'
folders = os.listdir('save')

commands = []

# go through each folder in '/save'
for folder in folders:
    # check if the folder name starts with 'CIFAR_5_clusters'
    if folder.startswith('CIFAR_5_clusters'):
        # check if folder contains a 'CIFAR_acc_matrix.pkl' file
        if 'CIFAR_acc_matrix.pkl' not in os.listdir('save/' + folder):
            # Initialize an empty dictionary to store the file content
            metadata = {}

            # Open the file and read line by line
            with open('save/' + folder + '/metadata.txt', 'r') as file:
                for line in file:
                    # Strip white space and split by colon
                    key, value = line.strip().split(': ')
                    # Convert numerical values from strings
                    if value.replace('.', '', 1).isdigit():
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    # Add to dictionary
                    metadata[key] = value

            # if metadata has key --runtime
            if 'runtime' in metadata: # means run is done
                # if it does, add a command to run test to the commands list
                commands.append('python3 test_CIFAR.py --experiment ' + folder)

print('length of commands:', len(commands))

# execute the commands
for i, command in enumerate(commands):
    subprocess.run(command, shell=True)

# go through each folder in '/save' and save each experiment's 'CIFAR_acc_matrix.pkl' to a pandas df
df = pandas.DataFrame()
for folder in folders:
    # check if the folder name starts with 'CIFAR_5_clusters'
    if folder.startswith('CIFAR_5_clusters'):
        # check if folder contains a 'CIFAR_acc_matrix.pkl' file
        if 'CIFAR_acc_matrix.pkl' in os.listdir('save/' + folder):
            # if it does, load the file and save it to a pandas df
            with open('save/' + folder + '/CIFAR_acc_matrix.pkl', 'rb') as f:
                acc_matrix = pickle.load(f)
            mean_matrix = acc_matrix.mean(axis=2)
            std_matrix = acc_matrix.std(axis=2)
            # add new row to the pandas df
            within_cluster_acc = mean_matrix.diagonal().mean()
            between_cluster_acc = mean_matrix[~np.eye(mean_matrix.shape[0],dtype=bool)].mean()
            cluster_1_acc = mean_matrix[0, 0]
            cluster_2_acc = mean_matrix[1, 1]
            cluster_3_acc = mean_matrix[2, 2]
            cluster_4_acc = mean_matrix[3, 3]
            cluster_5_acc = mean_matrix[4, 4]
            df = df._append({'experiment': folder, 'within_cluster_acc': within_cluster_acc, 
                             'between_cluster_acc': between_cluster_acc,
                             'cluster_1_acc': cluster_1_acc,
                             'cluster_2_acc': cluster_2_acc,
                             'cluster_3_acc': cluster_3_acc,
                             'cluster_4_acc': cluster_4_acc,
                             'cluster_5_acc': cluster_5_acc,}, ignore_index=True)

print(df.head())


# dump the pandas df to a pickle file
with open('5_cluster_testdata.pkl', 'wb') as f:
    pickle.dump(df, f)