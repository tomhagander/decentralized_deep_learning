from utils.testing_utils import test_on_double_MNIST
import argparse
import pickle

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

    if verbose:
        print('Testing double_MNIST on clients from experiment: {}'.format(args.experiment))
        print('Loading clients from: {}'.format(path))
    
    # import clients from clients.pkl
    with open(path, 'rb') as f:
        clients = pickle.load(f)

    if verbose:
        print('Clients loaded')
    
    acc_matrix = test_on_double_MNIST(clients, quick=quick, verbose=verbose)

    if verbose:
        print('Testing complete')

    with open('save/'+args.experiment+'/double_MNIST_acc_matrix.pkl', 'wb') as f:
        pickle.dump(acc_matrix, f)