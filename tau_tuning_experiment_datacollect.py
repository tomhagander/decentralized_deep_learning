
import pickle

# list of relevant experiment names (only seed 2 for label and seed 3 for 5 cluster) (aggregation based on volume of data)
label_invloss_names = ['CIFAR_DAC_invloss_tau_1_seed_2',
                    'CIFAR_DAC_invloss_tau_10_seed_2',
                    'CIFAR_DAC_invloss_tau_30_seed_2',
                    'CIFAR_label_DAC_invloss_tau_5_seed_2']

label_l2_expnames = ['CIFAR_label_DAC_l2_tau_1_seed_2',
                     'CIFAR_label_DAC_l2_tau_10_seed_2',
                     'CIFAR_label_DAC_l2_tau_80_seed_2',
                    'CIFAR_label_DAC_l2_tau_200_seed_2',
                    'CIFAR_label_DAC_l2_tau_30_seed_2',
                    ]

label_cosine_expnames = ['CIFAR_DAC_cosine_tau_1_seed_2',
                         'CIFAR_DAC_cosine_tau_10_seed_2',
                         'CIFAR_DAC_cosine_tau_30_seed_2',
                         'CIFAR_label_DAC_cosine_tau_80_seed_2',
                         'CIFAR_label_DAC_cosine_tau_200_seed_2', 
                         'CIFAR_label_DAC_cosine_tau_300_seed_2',
                         'CIFAR_label_DAC_cosine_tau_500_seed_2']

label_origin_expnames = ['CIFAR_label_DAC_cosine_origin_tau_1_seed_2',
                         'CIFAR_label_DAC_cosine_origin_tau_10_seed_2',
                         'CIFAR_label_DAC_cosine_origin_tau_30_seed_2',
                         'CIFAR_label_DAC_origin_tau_30_seed_2', # same???
                         'CIFAR_label_DAC_origin_tau_80_seed_2',
                         'CIFAR_label_DAC_origin_tau_200_seed_2', 
                         'CIFAR_label_DAC_cosine_origin_tau_300_seed_2',
                         'CIFAR_label_DAC_cosine_origin_tau_500_seed_2']

fivecluster_invloss_expnames = ['CIFAR_5_clusters_DAC_invloss_tau_1_seed_3',
                                'CIFAR_5_clusters_DAC_invloss_tau_5_seed_3',
                                'CIFAR_5_clusters_DAC_invloss_tau_10_seed_3',
                                'CIFAR_5_clusters_DAC_invloss_tau_30_seed_3',]

fivecluster_l2_expnames = ['CIFAR_5_clusters_DAC_l2_tau_1_seed_3',
                           'CIFAR_5_clusters_DAC_l2_tau_10_seed_3',
                           'CIFAR_5_clusters_DAC_l2_tau_30_seed_3',
                           'CIFAR_5_clusters_DAC_l2_tau_80_seed_3',
                           'CIFAR_5_clusters_DAC_l2_tau_200_seed_3']

fivecluster_cosine_expnames = ['CIFAR_5_clusters_DAC_cosine_tau_10_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_tau_30_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_tau_80_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_tau_200_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_tau_300_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_tau_500_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_tau_700_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_tau_1000_seed_3', 
                               'CIFAR_5_clusters_DAC_cosine_tau_2000_seed_3', 
                               'CIFAR_5_clusters_DAC_cosine_tau_5000_seed_3', 
                               'CIFAR_5_clusters_DAC_cosine_tau_10000_seed_3']

fivecluster_origin_expnames = ['CIFAR_5_clusters_DAC_cosine_origin_tau_1_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_origin_tau_10_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_origin_tau_30_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_origin_tau_80_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_origin_tau_200_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_origin_tau_300_seed_3',
                               'CIFAR_5_clusters_DAC_cosine_origin_tau_500_seed_3']

label_priorweight_invloss_expnames = ['CIFAR_label_DAC_priorweight_invloss_tau_1_seed_2',
                                      'CIFAR_label_DAC_priorweight_invloss_tau_5_seed_2',
                                      'CIFAR_label_DAC_priorweight_invloss_tau_10_seed_2',
                                      'CIFAR_label_DAC_priorweight_invloss_tau_30_seed_2']

label_priorweight_l2_expnames = ['CIFAR_label_DAC_priorweight_l2_tau_1_seed_2',
                                 'CIFAR_label_DAC_priorweight_l2_tau_5_seed_2',
                                 'CIFAR_label_DAC_priorweight_l2_tau_10_seed_2',
                                 'CIFAR_label_DAC_priorweight_l2_tau_30_seed_2',
                                    'CIFAR_label_DAC_priorweight_l2_tau_80_seed_2',
                                    'CIFAR_label_DAC_priorweight_l2_tau_200_seed_2']

label_priorweight_cosine_expnames = ['CIFAR_label_DAC_priorweight_cosine_tau_80_seed_2',
                                     'CIFAR_label_DAC_priorweight_cosine_tau_200_seed_2',
                                     'CIFAR_label_DAC_priorweight_cosine_tau_300_seed_2',]

label_priorweight_origin_expnames = ['CIFAR_label_DAC_priorweight_cosine_origin_tau_80_seed_2',
                                     'CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_2',
                                     'CIFAR_label_DAC_priorweight_cosine_origin_tau_300_seed_2',]

fivecluster_priorweight_invloss_expnames = ['CIFAR_5_clusters_DAC_priorweight_invloss_tau_1_seed_3',
                                            'CIFAR_5_clusters_DAC_priorweight_invloss_tau_5_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_invloss_tau_10_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_invloss_tau_30_seed_3']

fivecluster_priorweight_l2_expnames = ['CIFAR_5_clusters_DAC_priorweight_l2_tau_1_seed_3',
                                       'CIFAR_5_clusters_DAC_priorweight_l2_tau_5_seed_3',
                                       'CIFAR_5_clusters_DAC_priorweight_l2_tau_10_seed_3',
                                       'CIFAR_5_clusters_DAC_priorweight_l2_tau_30_seed_3',
                                    'CIFAR_5_clusters_DAC_priorweight_l2_tau_80_seed_3',
                                    'CIFAR_5_clusters_DAC_priorweight_l2_tau_200_seed_3']

fivecluster_priorweight_cosine_expnames = ['CIFAR_5_clusters_DAC_priorweight_cosine_tau_30_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_tau_80_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_tau_200_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_tau_300_seed_3', 
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_tau_500_seed_3']

fivecluster_priorweight_origin_expnames = ['CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_30_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_80_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_200_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_300_seed_3',
                                        'CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_500_seed_3']

def get_average_best_val_acc(clients):
    # for client in clients, get the best validation accuracy and add it to a list
    best_val_accs = []
    for client in clients:
        best_val_acc = max(client.val_acc_list)
        best_val_accs.append(best_val_acc)

    average_best_val_acc = sum(best_val_accs) / len(best_val_accs)
    return average_best_val_acc

def get_average_best_loss(clients):
    # for client in clients, get the lowest validation accuracy and add it to a list
    best_losses = []
    for client in clients:
        best_loss = min(client.val_loss_list)
        best_losses.append(best_loss)

    average_best_loss = sum(best_losses) / len(best_losses)
    return average_best_loss

def get_best_round_val_acc(clients):
    best_avg = 0
    for i in range(300):
        round_val_accs = []
        for client in clients:
            round_val_accs.append(client.val_acc_list[i])
        avg = sum(round_val_accs) / len(round_val_accs)
        if avg > best_avg:
            best_avg = avg
    return best_avg

def collect_experiments_and_save(expnames, results, toy=False):

    for exp in expnames:
        path = 'save/' + exp + '/'
        metadata = {}
        # also make sure the experiment has finished

        try:
            # Open the file and read line by line
            with open(path + 'metadata.txt', 'r') as file:
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

            # check if runtime exists
            if 'runtime' not in metadata:
                print(exp, ' runtime not found')
                continue
                #raise FileNotFoundError(exp + ' runtime not found')
            # check if path exists
        except:
            print(exp, ' metadata not found')
            continue
            #raise FileNotFoundError(exp + ' metadata not found')

        try:
            with open(path + 'clients.pkl', 'rb') as f:

                clients = pickle.load(f)

                if toy:
                    average_best_val_acc = get_average_best_loss(clients)
                else:
                    average_best_val_acc = get_average_best_val_acc(clients)
                #average_best_val_acc = get_best_round_val_acc(clients)
                print(exp, average_best_val_acc)

                # delete clients to free up memory
                del clients

                # from the metadatafile get the value of tau
                tau = metadata['tau']
                print('tau:', tau)

                # add tuple to results if not already in there
                if (tau, average_best_val_acc) not in results:
                    results.append((tau, average_best_val_acc))

        except:
            print(exp, ' not found')
            #raise FileNotFoundError(exp + ' not found')
            results.append((0,0))

    return results

def load_run_and_save(collection_of_expnames, resultsfile, toy=False):

    # load label_trainingweight_results from .pkl if it exists
    try:
        with open(resultsfile, 'rb') as f:
            results = pickle.load(f)
    except:
        results = [[], [], [], []]

    for i, expnames in enumerate(collection_of_expnames):
        results[i] = collect_experiments_and_save(expnames, results[i], toy=toy)
        print(expnames)
        print(results[i])

    # dump label_trainingweight_results to .pkl
    with open(resultsfile, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':

    label_trainingweight_expnames_collection = [label_invloss_names, label_l2_expnames, label_cosine_expnames, label_origin_expnames]
    fivecluster_trainingweight_expnames_collection = [fivecluster_invloss_expnames, fivecluster_l2_expnames, fivecluster_cosine_expnames, fivecluster_origin_expnames]
    label_priorweight_expnames_collection = [label_priorweight_invloss_expnames, label_priorweight_l2_expnames, label_priorweight_cosine_expnames, label_priorweight_origin_expnames]
    fivecluster_priorweight_expnames_collection = [fivecluster_priorweight_invloss_expnames, fivecluster_priorweight_l2_expnames, fivecluster_priorweight_cosine_expnames, fivecluster_priorweight_origin_expnames]

    load_run_and_save(fivecluster_trainingweight_expnames_collection, 'fivecluster_trainingweight_results.pkl')
    load_run_and_save(label_trainingweight_expnames_collection, 'label_trainingweight_results.pkl')
    load_run_and_save(label_priorweight_expnames_collection, 'label_priorweight_results.pkl')
    load_run_and_save(fivecluster_priorweight_expnames_collection, 'fivecluster_priorweight_results.pkl')
