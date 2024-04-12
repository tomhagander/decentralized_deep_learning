from utils.testing_utils import test_on_CIFAR
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='None', help='Experiment name')
    parser.add_argument('--quick', type=bool, default=False, help='Quick test, not implemented')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')
    args = parser.parse_args()

    quick = args.quick
    verbose = args.verbose

    path = 'save/' + args.experiment + '/clients.pkl'

    if verbose:
        print('Testing CIFAR-10 on clients from experiment: {}'.format(args.experiment))
        print('Loading clients from: {}'.format(path))
    # import clients from clients.pkl
    import pickle
    with open(path, 'rb') as f:
        clients = pickle.load(f)
    
    if verbose:
        print('Clients loaded')

    V_within, A_within, V_on_A, A_on_V = test_on_CIFAR(clients, quick=quick, verbose = verbose)
    
    if verbose:
        print('Testing complete')
    
    #save acc_matrix to pickle file called CIFAR_acc_matrix.pkl in save/experiment_name

    with open('save/'+args.experiment+'/CIFAR_V_within.pkl', 'wb') as f:
        pickle.dump(V_within, f)
    
    with open('save/'+args.experiment+'/CIFAR_A_within.pkl', 'wb') as f:
        pickle.dump(A_within, f)
    
    with open('save/'+args.experiment+'/CIFAR_V_on_A.pkl', 'wb') as f:
        pickle.dump(V_on_A, f)
    
    with open('save/'+args.experiment+'/CIFAR_A_on_V.pkl', 'wb') as f:
        pickle.dump(A_on_V, f)
    