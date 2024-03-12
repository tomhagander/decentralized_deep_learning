from utils.testing_utils import test_on_PACS
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='None', help='Experiment name')
    parser.add_argument('--quick', action='store_true', help='Set quick to true')
    args = parser.parse_args()

    quick = args.quick

    path = 'save/' + args.experiment + '/clients.pkl'

    # import clients from clients.pkl
    import pickle
    with open(path, 'rb') as f:
        clients = pickle.load(f)

    acc_matrix = test_on_PACS(clients, quick=True)
    
    #save acc_matrix to pickle file called PACS_acc_matrix.pkl in save/experiment_name

    with open('save/'+args.experiment+'/PACS_acc_matrix.pkl', 'wb') as f:
        pickle.dump(acc_matrix, f)
    