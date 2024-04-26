import os
import re
import shutil
import pickle


if __name__ == '__main__':

    # find all maps in save that have pacs in their name and print them
   
    save_path = 'save'
    all_files = os.listdir(save_path)
    pacs_files = [f for f in all_files if 'PACS' in f]

    # if that the file in pacs_files contains PACS_acc_matrix.pkl, then deleate the best model of each client so for client in clients: client.bestmodel = None
    # the clients are in the clients.pkl file for each folder in pacs_files so open the the pickel file and set the best model to None and overwrite the clients.pkl file with the same clients but with None as best model
    for pacs_file in pacs_files:
        pacs_path = os.path.join(save_path, pacs_file)
        client_file = 'clients.pkl'
        # if Pacs_acc_matrix.pkl is in the save/PACS..., delete the best model
        if 'PACS_acc_matrix.pkl' in os.listdir(pacs_path): 
            client_path = os.path.join(pacs_path, client_file)
            print(client_path)
            with open(client_path, 'rb') as f:
                clients = pickle.load(f)
            for client in clients:
                print('deleting client {} best model and local model'.format(client.idx))
                client.best_model = None
                client.local_model = None
            with open(client_path, 'wb') as f:
                pickle.dump(clients, f)
            print(f'best model of {pacs_path} is deleted')
