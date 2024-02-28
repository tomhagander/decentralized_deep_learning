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
from utils.training_utils import *
from utils.visualization_utils import *

from models.cifar_models import simple_CNN


# for now we use cifar10 and dont implement possibility to change dataset
# we can change models, and similiarity metrics, as well as other hyperparameters

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
                        stopping_rounds=args.stopping_rounds)
        clients.append(client)

    # training
    clients = train_clients_locally(clients, args.nbr_local_epochs, verbose=True)
    for round in range(args.nbr_rounds):
        # information exchange

        if args.client_information_exchange == 'DAC':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                      'prior_update_rule': args.prior_update_rule,
                      'similarity_metric': args.similarity_metric,
                      'tau': args.tau,
                      'cosine_alpha': args.cosine_alpha,
                      }
            clients = client_information_exchange_DAC(clients, 
                                            parameters=parameters,
                                            verbose=True,
                                            round=round)
        elif args.client_information_exchange == 'oracle':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled}
            clients = client_information_exchange_oracle(clients, 
                                            parameters=parameters,
                                            verbose=True,
                                            round=round,
                                            delusion=args.delusion)
            
        # validate post exchange and save to each clients val_losses_post_exchange and val_accs_post_exchange
        for client in clients:
            val_loss, val_acc = client.validate(client.local_model, train_set = False)
            client.val_losses_post_exchange.append(val_loss)
            client.val_accs_post_exchange.append(val_acc)
        
        # print client validation accuracy and loss
        val_accs = [client.val_accs_post_exchange[-1] for client in clients]
        val_losses = [client.val_losses_post_exchange[-1] for client in clients]
        print('Round {} post exchange. Average val acc: {:.3f}, average val loss: {:.3f}'.format(round, np.mean(val_accs), np.mean(val_losses)))
        # local training
        clients = train_clients_locally(clients, args.nbr_local_epochs, verbose=True)
        # print average client validation accuracy and loss§
        val_accs = [client.val_acc_list[-1] for client in clients]
        val_losses = [client.val_loss_list[-1] for client in clients]
        print('Round {} post local. Average val acc: {:.3f}, average val loss: {:.3f}'.format(round, np.mean(val_accs), np.mean(val_losses)))

    # done with training
    print('Done with training')

    # dump the clients to clients.pkl
    with open('save/'+results_folder+'/clients.pkl', 'wb') as f:
        pickle.dump(clients, f)
        f.close()

    # create ipynb copy of analyze_data.ipynb in folder
    import shutil
    shutil.copy('analyze_data.ipynb', 'save/'+results_folder+'/analyze_'+args.experiment_name+'.ipynb')

    