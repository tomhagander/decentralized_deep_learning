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
from utils.initialization_utils import sample_cifargroups, load_pacs, uniform_split
from utils.training_utils import train_clients_locally
from utils.training_utils import *
from utils.visualization_utils import *
from utils.initialization_utils import set_seed

from models.cifar_models import simple_CNN


if __name__ == '__main__':
    args = args_parser()

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

    # create ipynb copy of analyze_data.ipynb in folder
    import shutil
    shutil.copy('analyze_data.ipynb', 'save/'+results_folder+'/analyze_'+args.experiment_name+'.ipynb')

    # set number of classes and channels
    if args.dataset == 'PACS':
        args.nbr_classes = 7
        args.nbr_channels = 3

    # load data
    if args.dataset == 'PACS':
        # check if args.nbr_clients is divisible by 4, throw error otherwise
        if args.nbr_clients % 4 != 0:
            raise ValueError('PACS dataset requires number of clients to be divisible by 4')
        client_train_datasets, val_sets, test_sets = load_pacs('./PACS/', args.batch_size, args.nbr_clients // 4, augment=True)

    if args.dataset == 'PACS': # pretrained resnet18
        client_model_init = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        client_model_init.fc = torch.nn.Linear(client_model_init.fc.in_features, args.nbr_classes)

    # create clients
    clients = []
    if args.dataset == 'PACS':
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

    # keep the first two and the 30th and 31st client
    clients = clients[:2] + clients[30:32]

    photo_a = []
    sketch_a = []
    mixed_1_a = []
    mixed_2_a = []
    mixed_3_a = []
    mixed_4_a = []

    photo_b = []
    sketch_b = []
    mixed_1_b = []
    mixed_2_b = []
    mixed_3_b = []
    mixed_4_b = []

    def cosine_similarity(ci, cj):
        return np.dot(ci,cj)/(np.linalg.norm(ci)*np.linalg.norm(cj))

    # train clients locally
    clients = train_clients_locally(clients, args.nbr_local_epochs, verbose=True)
    for round in range(args.nbr_rounds):

        c0_a = clients[0].get_grad_a()
        c1_a = clients[1].get_grad_a()
        c2_a = clients[2].get_grad_a()
        c3_a = clients[3].get_grad_a()
        c0_b = clients[0].get_grad_b()
        c1_b = clients[1].get_grad_b()
        c2_b = clients[2].get_grad_b()
        c3_b = clients[3].get_grad_b()

        photo_a.append(cosine_similarity(c0_a, c1_a))
        photo_b.append(cosine_similarity(c0_b, c1_b))

        sketch_a.append(cosine_similarity(c2_a, c3_a))
        sketch_b.append(cosine_similarity(c2_b, c3_b))

        mixed_1_a.append(cosine_similarity(c0_a, c2_a))
        mixed_1_b.append(cosine_similarity(c0_b, c2_b))

        mixed_2_a.append(cosine_similarity(c0_a, c3_a))
        mixed_2_b.append(cosine_similarity(c0_b, c3_b))

        mixed_3_a.append(cosine_similarity(c1_a, c2_a))
        mixed_3_b.append(cosine_similarity(c1_b, c2_b))

        mixed_4_a.append(cosine_similarity(c1_a, c3_a))
        mixed_4_b.append(cosine_similarity(c1_b, c3_b))
        
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

        cosines = (photo_a, photo_b, sketch_a, sketch_b, mixed_1_a, mixed_1_b, mixed_2_a, mixed_2_b, mixed_3_a, mixed_3_b, mixed_4_a, mixed_4_b)
        with open('save/'+results_folder+'/cosines.pkl', 'wb') as f:
            pickle.dump(cosines, f)
            f.close()

    # done with training
    print('Done with training')
