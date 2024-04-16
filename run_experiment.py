import torch
import numpy as np
import random
import torchvision
from torchvision import transforms
from torchvision.models.resnet import ResNet18_Weights
import os
import pickle

from utils.classes import Client
from utils.arg_parser import args_parser
from utils.initialization_utils import sample_cifargroups, load_pacs, uniform_split, sample_cifargroups_5clusters
from utils.training_utils import train_clients_locally
from utils.training_utils import *
from utils.visualization_utils import *
from utils.initialization_utils import set_seed

from models.cifar_models import simple_CNN

import sys
sys.setrecursionlimit(200)


# for now we use cifar10 and dont implement possibility to change dataset
# we can change models, and similiarity metrics, as well as other hyperparameters

if __name__ == '__main__':
    # measure run time
    import time
    start_time = time.time()

    # parse arguments
    args = args_parser()

    print('Starting ', args.experiment_name)
    # set random seed
    set_seed(args.seed)

    # Set the device to use
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    # create folder for results
    results_folder = args.experiment_name
    if not os.path.exists('save/'+results_folder):
        # create folder in folder save
        os.makedirs('save/'+results_folder)

    # save args to metadata file
    with open('save/'+results_folder+'/metadata.txt', 'w') as f:
        for arg in vars(args):
            f.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        f.close()

    # mergatron data
    mergatron_stops = np.zeros(args.nbr_rounds)

    # create ipynb copy of analyze_data.ipynb in folder
    import shutil
    shutil.copy('analyze_data.ipynb', 'save/'+results_folder+'/analyze_'+args.experiment_name+'.ipynb')

    # set number of classes and channels
    if args.dataset == 'cifar10':
        args.nbr_classes = 10
        args.nbr_channels = 3
    elif args.dataset == 'PACS':
        args.nbr_classes = 7
        args.nbr_channels = 3

    # load dataset and transform
    if args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10('.', train=True, download=True, transform=trans_cifar)

        if args.shift == 'label':
            # assign data to clients (label shift, animals and vehicles)
            dict_users, dict_users_val = sample_cifargroups(train_dataset, args.nbr_clients, args.n_data_train, args.n_data_val, args.CIFAR_ratio)
            # dicts contain indices of data for each client
        elif args.shift == '5_clusters':
            args.CIFAR_ratio = 0.2
            dict_users, dict_users_val = sample_cifargroups_5clusters(train_dataset, args.nbr_clients, args.n_data_train, args.n_data_val, args.CIFAR_ratio)
        elif args.shift == 'PANM_swap':
            dict_users, dict_users_val = uniform_split(train_dataset, args.nbr_clients, args.n_data_train, args.n_data_val)
            
            # if client belongs to cluster 0, swap labels 0 and 1
            # if client belongs to cluster 1, swap labels 6 and 7

            for i in range(args.nbr_clients):
                if i < args.nbr_clients * args.CIFAR_ratio:
                    # cluster 0
                    for j in range(len(dict_users[i])):
                        if train_dataset.targets[dict_users[i][j]] == 0:
                            train_dataset.targets[dict_users[i][j]] = 1
                        elif train_dataset.targets[dict_users[i][j]] == 1:
                            train_dataset.targets[dict_users[i][j]] = 0
                else:
                    # cluster 1
                    for j in range(len(dict_users[i])):
                        if train_dataset.targets[dict_users[i][j]] == 6:
                            train_dataset.targets[dict_users[i][j]] = 7
                        elif train_dataset.targets[dict_users[i][j]] == 7:
                            train_dataset.targets[dict_users[i][j]] = 6

        elif args.shift == 'PANM_swap4':
            dict_users, dict_users_val = uniform_split(train_dataset, args.nbr_clients, args.n_data_train, args.n_data_val)
            
            # if client belongs to cluster 0, swap labels 0 and 1
            # if client belongs to cluster 1, swap labels 2 and 3
            # if client belongs to cluster 2, swap labels 4 and 5
            # if client belongs to cluster 3, swap labels 6 and 7

            for i in range(args.nbr_clients):
                if i < args.nbr_clients * args.CIFAR_ratio:
                    # cluster 0
                    for j in range(len(dict_users[i])):
                        if train_dataset.targets[dict_users[i][j]] == 0:
                            train_dataset.targets[dict_users[i][j]] = 1
                        elif train_dataset.targets[dict_users[i][j]] == 1:
                            train_dataset.targets[dict_users[i][j]] = 0
                elif i < 2 * args.nbr_clients * args.CIFAR_ratio:
                    # cluster 1
                    for j in range(len(dict_users[i])):
                        if train_dataset.targets[dict_users[i][j]] == 2:
                            train_dataset.targets[dict_users[i][j]] = 3
                        elif train_dataset.targets[dict_users[i][j]] == 3:
                            train_dataset.targets[dict_users[i][j]] = 2
                elif i < 3 * args.nbr_clients * args.CIFAR_ratio:
                    # cluster 2
                    for j in range(len(dict_users[i])):
                        if train_dataset.targets[dict_users[i][j]] == 4:
                            train_dataset.targets[dict_users[i][j]] = 5
                        elif train_dataset.targets[dict_users[i][j]] == 5:
                            train_dataset.targets[dict_users[i][j]] = 4
                else:
                    # cluster 3
                    for j in range(len(dict_users[i])):
                        if train_dataset.targets[dict_users[i][j]] == 6:
                            train_dataset.targets[dict_users[i][j]] = 7
                        elif train_dataset.targets[dict_users[i][j]] == 7:
                            train_dataset.targets[dict_users[i][j]] = 6
        else:
            pass


    elif args.dataset == 'PACS':
        # check if args.nbr_clients is divisible by 4, throw error otherwise
        if args.nbr_clients % 4 != 0:
            raise ValueError('PACS dataset requires number of clients to be divisible by 4')
        client_train_datasets, val_sets, test_sets = load_pacs('./PACS/', args.batch_size, args.nbr_clients // 4, augment=True)


    # load model (same initialization for all clients)
    if args.dataset == 'cifar10': # custom cnn
        client_model_init = simple_CNN(nbr_classes=args.nbr_classes)
    elif args.dataset == 'PACS': # pretrained resnet18
        # client_model_init = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT) # change here for pretrained
        client_model_init = torchvision.models.resnet18(weights=None) # change here for Not pretrained
        client_model_init.fc = torch.nn.Linear(client_model_init.fc.in_features, args.nbr_classes)

    # create clients
    clients = []
    if args.dataset == 'cifar10':
        for i in range(args.nbr_clients):
            print('creating client {}'.format(i))
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
                            stopping_rounds=args.stopping_rounds,
                            ratio = args.CIFAR_ratio,
                            dataset = 'cifar10',
                            shift=args.shift)
            
            clients.append(client)
    elif args.dataset == 'PACS':
        for i in range(args.nbr_clients):
            print('creating client {}'.format(i))
            # choose val_set from val_sets based on i
            if (i<args.nbr_clients//4):
                val_set = val_sets[0]
            elif (i<args.nbr_clients//2):
                val_set = val_sets[1]
            elif (i<3*args.nbr_clients//4):
                val_set = val_sets[2]
            else:
                val_set = val_sets[3]
            client = Client(train_set=client_train_datasets[i], 
                            val_set=val_set, 
                            idxs_train=None, 
                            idxs_val=None, 
                            criterion=torch.nn.CrossEntropyLoss(), 
                            lr=args.lr, 
                            device=device, 
                            batch_size=args.batch_size, 
                            num_users=args.nbr_clients, 
                            model=client_model_init,
                            idx=i,
                            stopping_rounds=args.stopping_rounds,
                            ratio = 0.25,
                            dataset = 'PACS',
                            shift=args.shift)
            clients.append(client)

    if args.client_information_exchange == 'some_delusion': #ONLY for some_delusion (Ignore)
        delusional_client_idxs = get_delusional_clients(clients, args.nbr_deluded_clients) 
    
    # training
    clients = train_clients_locally(clients, args.nbr_local_epochs, verbose=True)

    # measure all similarities if flag is set
    if args.measure_all_similarities:
        for client in clients:
            client.measure_all_similarities(clients, args.similarity_metric)
    
    for round in range(args.nbr_rounds):
        # information exchange

        if args.client_information_exchange == 'DAC':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                      'prior_update_rule': args.prior_update_rule,
                      'similarity_metric': args.similarity_metric,
                      'tau': args.tau,
                      'cosine_alpha': args.cosine_alpha,
                      'mergatron': args.mergatron,
                      }
            clients = client_information_exchange_DAC(clients, 
                                            parameters=parameters,
                                            verbose=True,
                                            round=round)
        elif args.client_information_exchange == 'oracle':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                          'mergatron': args.mergatron,
                          }
            clients = client_information_exchange_oracle(clients, 
                                            parameters=parameters,
                                            verbose=True,
                                            round=round,
                                            delusion=args.delusion)
        elif args.client_information_exchange == 'PANM':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                      'similarity_metric': args.similarity_metric,
                      'NAEM_frequency': args.NAEM_frequency,
                      'T1': args.T1,
                      'cosine_alpha': args.cosine_alpha,
                      }
            clients = client_information_exchange_PANM(clients, 
                                                       parameters=parameters, 
                                                       verbose = True, 
                                                       round = round)
        elif args.client_information_exchange == 'no_exchange':
            pass
        elif args.client_information_exchange == 'some_delusion':

            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                          'start_delusion': 15,
                          'delusional_client_idxs': delusional_client_idxs}
            clients = client_information_exchange_some_delusion(clients, 
                                                                parameters=parameters, 
                                                                verbose = True, 
                                                                round = round)
            
        elif args.client_information_exchange == 'look_hard_once':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                      'similarity_metric': args.similarity_metric,
                      'cosine_alpha': args.cosine_alpha,
                      }
            clients = client_information_exchange_LHO(clients, 
                                            parameters=parameters,
                                            verbose=True,
                                            round=round)
        
        
        # validate post exchange and save to each clients val_losses_post_exchange and val_accs_post_exchange
        for client in clients:
            val_loss, val_acc = client.validate(client.local_model, train_set = False)

            if args.mergatron == 'activate':
                last_acc = client.val_acc_list[-1]
                if val_acc < last_acc: # cancel merge
                    print('Mergatron denies merge!')
                    print('Client {} did not improve after merge, reverting to previous model'.format(client.idx))
                    client.local_model.load_state_dict(client.pre_merge_model)
                    mergatron_stops[round] += 1

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

        # measure all similarities if flag is set
        if args.measure_all_similarities:
            for client in clients:
                client.measure_all_similarities(clients, args.similarity_metric)

        # dump the clients to clients.pkl
        with open('save/'+results_folder+'/clients.pkl', 'wb') as f:
            pickle.dump(clients, f)
            f.close()

    # done with training
    print('Done with training')

    # send all models back to cpu:
    for client in clients:
        client.local_model.to('cpu')
        client.best_model.to('cpu')

    # end time
    end_time = time.time()
    print('Runtime: {} = {} hours'.format(end_time - start_time, (end_time - start_time)/3600))
    # save time to metadata file
    with open('save/'+results_folder+'/metadata.txt', 'a') as f:
        f.write('runtime: {}'.format(end_time - start_time))
        f.close()

    # dump mergatron stops
    if args.mergatron == 'activate':
        with open('save/'+results_folder+'/mergatron_stops.pkl', 'wb') as f:
            pickle.dump(mergatron_stops, f)
            f.close()


    