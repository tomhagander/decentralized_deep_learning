from utils.testing_utils import test_on_CIFAR, test_5_clusters
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='None', help='Experiment name')
    parser.add_argument('--quick', type=bool, default=False, help='Quick test, not implemented')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')
    args = parser.parse_args()

    quick = args.quick
    verbose = args.verbose

    metadata_path = 'save/' + args.experiment + '/metadata.txt'
    # load metadata to dictionary
    # Initialize an empty dictionary to store the file content
    metadata = {}

    # Open the file and read line by line
    with open(metadata_path, 'r') as file:
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

    path = 'save/' + args.experiment + '/clients.pkl'

    if metadata['shift'] == 'label':
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
        
    elif metadata['shift'] == '5_clusters':
        if verbose:
            print('Testing CIFAR-10 on clients from experiment: {}'.format(args.experiment))
            print('Loading clients from: {}'.format(path))
        # import clients from clients.pkl
        import pickle
        with open(path, 'rb') as f:
            clients = pickle.load(f)
        
        if verbose:
            print('Clients loaded')

        acc_matrix = test_5_clusters(clients)

        # dump to pickle file
        with open('save/'+args.experiment+'/CIFAR_acc_matrix.pkl', 'wb') as f:
            pickle.dump(acc_matrix, f)

        