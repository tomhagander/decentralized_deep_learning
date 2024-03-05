import torch
import numpy as np
import random
import torchvision
from torchvision import transforms
import os
import pickle

from utils.classes import Client
from utils.arg_parser import args_parser
from utils.initialization_utils import sample_cifargroups
from utils.training_utils import train_clients_locally
from utils.training_utils import client_information_exchange_DAC
from utils.visualization_utils import *

from models.cifar_models import simple_CNN

# train a client locally to see that it works and to find good hyperparameters

if __name__ == '__main__':
    args = args_parser()

    # Set the device to use
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create folder for results
    results_folder = args.experiment_name
    if not os.path.exists('save/'+results_folder):
        # create folder in folder save
        os.makedirs('save/'+results_folder)
    
    if not os.path.exists('save/'+results_folder+'/figs'):
        os.makedirs('save/'+results_folder+'/figs')

    figpath = 'save/'+results_folder+'/figs/'

    # save args to metadata file
    with open('save/'+results_folder+'/metadata.txt', 'w') as f:
        f.write(str(args))
        # write that is was a local training
        f.write('\nLocal training')
        f.close()

    # load dataset and transform
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10('.', train=True, download=True, transform=trans_cifar)

    # assign data to clients (for now label shift, animals and vehicles)
    # TODO: covariate shift
    dict_users, dict_users_val = sample_cifargroups(train_dataset, args.nbr_clients, args.n_data_train, args.n_data_val)

    # dicts contain indices of data for each client

    # load model (same initialization for all clients)
    client_model_init = simple_CNN(nbr_classes=10)

    # create clients
    clients = []
    for i in range(args.nbr_clients):
        client = Client(train_set=train_dataset, 
                        idxs_train=dict_users[i], 
                        idxs_val=dict_users_val[i], 
                        criterion=torch.nn.CrossEntropyLoss(), 
                        lr=args.lr, 
                        device=device, 
                        batch_size=args.batch_size, 
                        num_users=args.nbr_clients, 
                        model=client_model_init,
                        idx=i,
                        stopping_rounds=args.stopping_rounds,)
        clients.append(client)


    # training
    for epoch in range(args.nbr_rounds):
        for client in clients:
            client.train(args.nbr_local_epochs)
            # print client validation accuracy and loss
            val_loss, val_acc = client.validate(client.local_model, train_set=False)
            print('Epoch {} client {}. Val acc: {:.3f}, val loss: {:.3f}'.format(epoch, client.idx, val_acc, val_loss))

    # done with training
    print('Done with training')

    #plotting
    plot_client_validation_and_training_loss(clients[0], figpath)
    plot_client_validation_accuracy(clients[0], figpath)
    plot_client_validation_and_training_loss(clients[1], figpath)
    plot_client_validation_accuracy(clients[1], figpath)

    # dump the clients to clients.pkl
    with open('save/'+results_folder+'/clients.pkl', 'wb') as f:
        pickle.dump(clients, f)
        f.close()

    # create ipynb copy of analyze_data.ipynb in folder
    import shutil
    shutil.copy('analyze_data.ipynb', 'save/'+results_folder+'/analyze_'+args.experiment_name+'.ipynb')

    