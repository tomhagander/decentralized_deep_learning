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
from utils.initialization_utils import sample_cifargroups, load_pacs
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

    # save args to metadata file
    with open('save/'+results_folder+'/metadata.txt', 'w') as f:
        for arg in vars(args):
            f.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        f.close()

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

        # assign data to clients (for now label shift, animals and vehicles)
        # TODO: covariate shift
        dict_users, dict_users_val = sample_cifargroups(train_dataset, args.nbr_clients, args.n_data_train, args.n_data_val, args.CIFAR_ratio)
        # dicts contain indices of data for each client
    elif args.dataset == 'PACS':
        # check if args.nbr_clients is divisible by 4, throw error otherwise
        if args.nbr_clients % 4 != 0:
            raise ValueError('PACS dataset requires number of clients to be divisible by 4')
        client_train_datasets, val_sets, test_sets = load_pacs('./PACS/', args.batch_size, args.nbr_clients // 4, augment=True)


    # load model (same initialization for all clients)
    if args.dataset == 'cifar10': # custom cnn
        client_model_init = simple_CNN(nbr_classes=args.nbr_classes)
    elif args.dataset == 'PACS': # pretrained resnet18
        client_model_init = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        client_model_init.fc = torch.nn.Linear(client_model_init.fc.in_features, args.nbr_classes)

    # create clients
    clients = []
    if args.dataset == 'cifar10':
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
                            stopping_rounds=args.stopping_rounds,
                            ratio = args.CIFAR_ratio,
                            dataset = 'cifar10')
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
                            dataset = 'PACS')
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
        elif args.client_information_exchange == 'PANM':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                      'similarity_metric': args.similarity_metric,
                      'NAEM_frequency': args.NAEM_frequency,
                      'T1': args.T1,
                      'cosine_alpha': args.cosine_alpha,
                      }
            clients = client_information_exchange_PANM(clients, parameters, verbose = True, round = round)
        elif args.client_information_exchange == 'no_exchange':
            pass
            
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

    