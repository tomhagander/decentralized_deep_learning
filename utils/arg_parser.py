import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, 0 for our GPU, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--shift', type=str, default='label', help="(type of shift)")
    parser.add_argument('--nbr_rounds', type=int, default=270, help="rounds of information exchange")
    parser.add_argument('--nbr_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--n_data_train', type=int, default=400, help="train size")
    parser.add_argument('--n_data_val', type=int, default=100, help="validation size")
    parser.add_argument('--seed', type=int, default= int.from_bytes(os.urandom(4), 'big'), help='random seed (default: random int)')
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--nbr_local_epochs', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--lr', type=float, default=7.5e-5, help="learning rate")
    parser.add_argument('--stopping_rounds', type=int, default=50, help='rounds of early stopping')
    parser.add_argument('--nbr_neighbors_sampled', type=int, default=5, help='number of neighbors sampled')
    parser.add_argument('--prior_update_rule', type=str, default='softmax-variable-tau', help='how to update priors')
    parser.add_argument('--similarity_metric', type=str, default='inverse_training_loss', help='how to measure similarity between clients')
    parser.add_argument('--cosine_alpha', type=float, default=0, help='alpha in cosine similarity')
    parser.add_argument('--tau', type=float, default=30, help='temperature in softmax')
    parser.add_argument('--client_information_exchange', type=str, default='DAC', help='How clients exchange information')
    parser.add_argument('--experiment_name', type=str, default='experiment', help='name of experiment')
    parser.add_argument('--delusion', type=float, default=0, help='If oracle, the chance that a client communicates with the wrong cluster in a round. If -1, communication is random')
    parser.add_argument('--NAEM_frequency', type = int, default=5, help='How often NAEM is performed')
    parser.add_argument('--T1', type=int, default=50, help='When to change from NSMS to NAEM')
    parser.add_argument('--CIFAR_ratio', type=float, default=0.4, help='Ratio of size of clients in vehicles cluster to animals cluster in CIFAR10')
    parser.add_argument('--nbr_deluded_clients', type=int, default=0, help='If comms is some_delusion, the number of clients that communicate with the wrong cluster in a round')
    parser.add_argument('--measure_all_similarities', type=bool, default=False, help='If True, all similarities are measured parallel to training and saved')
    parser.add_argument('--mergatron', type=str, default='chill', help='type activate to activate mergatron')
    parser.add_argument('--aggregation_weighting', type=str, default='trainset_size', help='The method of weighting aggregated models')
    parser.add_argument('--minmax', type=bool, default=False, help='If True, minmax scale the similarities')

    # wierdos dont change
    parser.add_argument('--nbr_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--nbr_channels', type=int, default=3, help="number of channels of images")
    # arguments from DAC
    '''
    parser.add_argument('--n_rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--n_rounds_pens', type=int, default=0, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=8, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    
    parser.add_argument('--a', type=float, default='0', help="a value in minmax scale")
    parser.add_argument('--b', type=float, default='1', help="b value in minmax scale")
    parser.add_argument('--tau', type=float, default='1', help="temperature in softmax")
    
    parser.add_argument('--n_sampled', type=int, default=5)
    parser.add_argument('--top_m', type=int, default=0)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--pens', action='store_true')
    parser.add_argument('--DAC', action='store_true')
    parser.add_argument('--DAC_var', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--random', action='store_true')
    
    parser.add_argument('--n_data_train', type=int, default=100, help="train size")
    parser.add_argument('--n_data_val', type=int, default=100, help="validation size")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--iid', type=str, default='covariate', help="covariate or label (type of shift)")
    #parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    '''
    args = parser.parse_args()
    return args