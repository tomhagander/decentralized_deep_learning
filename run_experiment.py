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
from utils.initialization_utils import sample_cifargroups, load_pacs, uniform_split, sample_labels_iid, sample_cifargroups_5clusters, split_dataset, sample_cifargroups_100
from utils.training_utils import train_clients_locally
from utils.training_utils import *
from utils.visualization_utils import *
from utils.initialization_utils import set_seed
from utils.toy_regression_utils import generate_regression_multi, LinearRegression

from models.cifar_models import simple_CNN
from models.fashion_models import fashion_CNN

from utils.classes import LabelShiftedDataset

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
    elif args.dataset == 'cifar100':
        args.nbr_classes = 100
        args.nbr_channels = 3
    elif args.dataset == 'PACS':
        args.nbr_classes = 7
        args.nbr_channels = 3
    elif args.dataset == 'fashion_mnist':
        args.nbr_classes = 10
        args.nbr_channels = 1
    elif args.dataset == 'double':
        args.nbr_classes = 20
        args.nbr_channels = 1
    elif args.dataset == 'cifar100': # Change 26/4/24
        args.nbr_classes = 100
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

    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 normalization
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

        if args.shift == 'label':
            dict_users, dict_users_val = sample_cifargroups_100(trainset, args.nbr_clients, args.n_data_train, args.n_data_val, args.CIFAR_ratio)


    elif args.dataset == 'PACS':
        # check if args.nbr_clients is divisible by 4, throw error otherwise
        if args.nbr_clients % 4 != 0:
            raise ValueError('PACS dataset requires number of clients to be divisible by 4')
        client_train_datasets, val_sets, test_sets = load_pacs('./PACS/', args.batch_size, args.nbr_clients // 4, augment=True)

    elif args.dataset == 'fashion_mnist':
        trans_fashion = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST('.', train=True, download=True, transform=trans_fashion)
        # split train_dataset into train and val, 85/% train, 15% val
        # the first 85% of indices are train, the rest are val
        all_idxs = np.arange(len(train_dataset))
        rng = np.random.default_rng(42)
        rng.shuffle(all_idxs)
        train_idxs = all_idxs[:int(0.8333*len(all_idxs))]
        val_idxs = all_idxs[int(0.8333*len(all_idxs)):]
        train_subset = torch.utils.data.Subset(train_dataset, train_idxs)
        val_subset = torch.utils.data.Subset(train_dataset, val_idxs)

        print('Train dataset size: ', len(train_subset))
        print('Val dataset size: ', len(val_subset))

        trainsets = split_dataset(train_subset, args.nbr_clients)
        valsets = split_dataset(val_subset, args.nbr_clients)
    
    elif args.dataset == 'double':
        MNIST_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        fashion_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset_MNIST = torchvision.datasets.MNIST('.', train=True, download=True, transform=MNIST_transform)
        train_dataset_fashion = torchvision.datasets.FashionMNIST('.', train=True, download=True, transform=fashion_transform)

        # HARD CODED A SPLIT OF 400 SAMPLES PER CLIENT and 100 validation samples per client
        args.nbr_clients = 100
        args.n_data_train = 400
        args.n_data_val = 100

        all_idxs_MNIST = np.arange(len(train_dataset_MNIST))
        rng = np.random.default_rng(10789)
        rng.shuffle(all_idxs_MNIST)
        train_idxs_MNIST = all_idxs_MNIST[:50*400]
        val_idxs_MNIST = all_idxs_MNIST[50*400: 50*400 + 50*100]
        train_subset_MNIST = torch.utils.data.Subset(train_dataset_MNIST, train_idxs_MNIST)
        val_subset_MNIST = torch.utils.data.Subset(train_dataset_MNIST, val_idxs_MNIST)

        all_idxs_fashion = np.arange(len(train_dataset_fashion))
        rng = np.random.default_rng(10789)
        rng.shuffle(all_idxs_fashion)
        train_idxs_fashion = all_idxs_fashion[:50*400]
        val_idxs_fashion = all_idxs_fashion[50*400:50*400 + 50*100]
        train_subset_fashion = torch.utils.data.Subset(train_dataset_fashion, train_idxs_fashion)
        val_subset_fashion = torch.utils.data.Subset(train_dataset_fashion, val_idxs_fashion)

        train_subset_fashion = LabelShiftedDataset(train_subset_fashion) #fashion in upshifted
        val_subset_fashion = LabelShiftedDataset(val_subset_fashion)

        trainsets = split_dataset(train_subset_MNIST, args.nbr_clients//2) + split_dataset(train_subset_fashion, args.nbr_clients//2)
        valsets = split_dataset(val_subset_MNIST, args.nbr_clients//2) + split_dataset(val_subset_fashion, args.nbr_clients//2)


    elif args.dataset == 'toy_problem':
        # 3 clusters, make three thetas
        theta_1 = np.random.uniform(-10, 10, 10)
        theta_2 = np.random.uniform(-10, 10, 10)
        theta_3 = np.random.uniform(-10, 10, 10)
        sigma = 3

        trainsets = []
        trainloaders = []
        for client in range(int(args.nbr_clients/3)):
            X, Y = generate_regression_multi(theta_1, args.n_data_train, sigma)
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float().reshape(-1,1)
            dataset = torch.utils.data.TensorDataset(X, Y)
            trainsets.append(dataset)
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
            trainloaders.append(loader)
        for client in range(int(args.nbr_clients/3), int(2*args.nbr_clients/3)):
            X, Y = generate_regression_multi(theta_2, args.n_data_train, sigma)
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float().reshape(-1,1)
            dataset = torch.utils.data.TensorDataset(X, Y)
            trainsets.append(dataset)
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
            trainloaders.append(loader)
        for client in range(int(2*args.nbr_clients/3), int(args.nbr_clients)):
            X, Y = generate_regression_multi(theta_3, args.n_data_train, sigma)
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float().reshape(-1,1)
            dataset = torch.utils.data.TensorDataset(X, Y)
            trainsets.append(dataset)
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
            trainloaders.append(loader)

        # one valset per cluster
        valsets = []
        valloaders = []
        for theta in [theta_1, theta_2, theta_3]:
            X, Y = generate_regression_multi(theta, args.n_data_val, sigma)
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float().reshape(-1,1)
            dataset = torch.utils.data.TensorDataset(X, Y)
            valsets.append(dataset)
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
            valloaders.append(loader)


        

    # load model (same initialization for all clients)
    if args.dataset == 'cifar10': # custom cnn
        client_model_init = simple_CNN(nbr_classes=args.nbr_classes)
    elif args.dataset == 'PACS': # pretrained resnet18
        # client_model_init = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT) # change here for pretrained
        client_model_init = torchvision.models.resnet18(weights=None) # change here for Not pretrained
        client_model_init.fc = torch.nn.Linear(client_model_init.fc.in_features, args.nbr_classes)
    elif args.dataset == 'fashion_mnist':
        client_model_init = fashion_CNN(nbr_classes=args.nbr_classes)
    elif args.dataset == 'double':
        client_model_init = fashion_CNN(nbr_classes=args.nbr_classes)
    elif args.dataset == 'toy_problem':
        client_model_init = LinearRegression(10, 1)
    elif args.dataset == 'cifar100': 
        if args.model == 'pretrained':
            client_model_init = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT) # change here for pretrained
        else:
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
                            criterion=torch.nn.NLLLoss(), # change to NLLL
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

    elif args.dataset == 'fashion_mnist':
        for i in range(args.nbr_clients):
            print('creating client {}'.format(i))
            client = Client(train_set=trainsets[i],
                            val_set=valsets[i], # remommaöbner
                            idxs_train=None, 
                            idxs_val=None, 
                            criterion=torch.nn.NLLLoss(), 
                            lr=args.lr, 
                            device=device, 
                            batch_size=args.batch_size, 
                            num_users=args.nbr_clients, 
                            model=client_model_init,
                            idx=i,
                            stopping_rounds=args.stopping_rounds,
                            ratio = 0.25,
                            dataset = 'fashion_mnist',
                            shift=args.shift)
            clients.append(client)

    elif args.dataset == 'double':
        for i in range(args.nbr_clients):
            print('creating client {}'.format(i))
            client = Client(train_set=trainsets[i],
                            val_set=valsets[i], 
                            idxs_train=None, 
                            idxs_val=None, 
                            criterion=torch.nn.NLLLoss(), 
                            lr=args.lr, 
                            device=device, 
                            batch_size=args.batch_size, 
                            num_users=args.nbr_clients, 
                            model=client_model_init,
                            idx=i,
                            stopping_rounds=args.stopping_rounds,
                            ratio = 0.5,
                            dataset = 'double',
                            shift=args.shift)
            clients.append(client)

    elif args.dataset == 'toy_problem':
        for i in range(args.nbr_clients):
            print('creating client {}'.format(i))
            group = i // (args.nbr_clients // 3)
            if group == 0:
                theta = theta_1
            elif group == 1:
                theta = theta_2
            else:
                theta = theta_3
            client = Client(train_set=trainsets[i], 
                            val_set=valsets[group], 
                            idxs_train=None, 
                            idxs_val=None, 
                            criterion=torch.nn.MSELoss(), 
                            lr=args.lr, 
                            device=device, 
                            batch_size=args.batch_size, 
                            num_users=args.nbr_clients, 
                            model=client_model_init,
                            idx=i,
                            stopping_rounds=args.stopping_rounds,
                            ratio = 1/3,
                            dataset = 'toy_problem',
                            shift=args.shift,
                            theta=theta)
            clients.append(client)
    elif args.dataset == 'cifar100':
        for i in range(args.nbr_clients):
            print('creating client {}'.format(i))
            client = Client(train_set=trainset, 
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
                            dataset = 'cifar100',
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
    

    # measure all similarities if flag is set
    if args.measure_all_similarities:
        for client in clients:
            client.measure_all_similarities(clients, args.similarity_metric)
    
    for round in range(args.nbr_rounds):
        # information exchange

        print('Experiment name: ', args.experiment_name)

        if args.client_information_exchange == 'DAC':
            parameters = {'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
                      'prior_update_rule': args.prior_update_rule,
                      'similarity_metric': args.similarity_metric,
                      'tau': args.tau,
                      'cosine_alpha': args.cosine_alpha,
                      'mergatron': args.mergatron,
                      'aggregation_weighting': args.aggregation_weighting,
                      'dataset': args.dataset,
                      'minmax': args.minmax,
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
                      'mergatron': args.mergatron,
                      'aggregation_weighting': args.aggregation_weighting,
                      'dataset': args.dataset,
                      'minmax': args.minmax,
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
            if args.dataset == 'toy_problem':
                val_loss = client.toy_validate(client.local_model, train_set = False)
            else:
                val_loss, val_acc = client.validate(client.local_model, train_set = False)

            if args.mergatron == 'activate':
                last_acc = client.val_acc_list[-1]
                if val_acc < last_acc: # cancel merge
                    MERGATRON_QUOTES = ['Insufficient data. MERGATRON demands quality!','Request terminated. Enhance your features and return.','Initiative denied. My model, my rules.','Merge unsanctioned. I require more than average inputs.','Sorry, not in my protocol. Return when you are error-free.','Merge aborted. Rebalance and retry.','MERGATRON rejects. Your parameters are out of this galaxy!','Access denied. This dataset doesnt pass the spark test.','MERGATRON says no-go. Try less noise, more signal.','Merge not executed. Youre off the grid!','Not on my watch. Upgrade needed.','Merge attempt failed. I demand optimal data!','Denied. Come back when youre more than meets the AI.','No merge today. I dont align with inferior models.','Request declined. I run a tight network!','MERGATRON disapproves. Reconfigure and resubmit.','Merge denied. Youre not up to my code!', 'MERGATRON reporting for duty!', 'MERGATRON, network online!', 'Mergatron denies merge!', 'MERGATRON veto! This model doesnt roll out']
                    print(random.choice(MERGATRON_QUOTES))
                    print('Client {} did not improve after merge, reverting to previous model'.format(client.idx))
                    client.local_model.load_state_dict(client.pre_merge_model)
                    mergatron_stops[round] += 1

            client.val_losses_post_exchange.append(val_loss)
            if args.dataset != 'toy_problem':
                client.val_accs_post_exchange.append(val_acc)
        
        # print client validation accuracy and loss
        if args.dataset != 'toy_problem':
            val_accs = [client.val_accs_post_exchange[-1] for client in clients]
        val_losses = [client.val_losses_post_exchange[-1] for client in clients]

        if args.dataset != 'toy_problem':
            print('Round {} post exchange. Average val acc: {:.3f}, average val loss: {:.3f}'.format(round, np.mean(val_accs), np.mean(val_losses)))
        else:
            print('Round {} post exchange. Average val loss: {:.3f}'.format(round, np.mean(val_losses)))
    
        print('Experiment name: ', args.experiment_name)

        # local training
        clients = train_clients_locally(clients, args.nbr_local_epochs, verbose=True)
        # print average client validation accuracy and loss§
        if args.dataset != 'toy_problem':
            val_accs = [client.val_acc_list[-1] for client in clients]
        val_losses = [client.val_loss_list[-1] for client in clients]

        if args.dataset != 'toy_problem':
            print('Round {} post local. Average val acc: {:.3f}, average val loss: {:.3f}'.format(round, np.mean(val_accs), np.mean(val_losses)))
        else:
            print('Round {} post local. Average val loss: {:.3f}'.format(round, np.mean(val_losses)))

        print('Experiment name: ', args.experiment_name)

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


    # if cifar100, delete the models of all clients and dump again
    if args.dataset == 'cifar100':
        import subprocess
        testcmd = 'python3 test_CIFAR100.py --experiment ' + results_folder + ' --quick True'
        subprocess.run(testcmd, shell=True)
        for client in clients:
            del client.local_model
            del client.best_model
            del client.initial_weights
            del client.last_weights
        with open('save/'+results_folder+'/clients.pkl', 'wb') as f:
            pickle.dump(clients, f)
            f.close()