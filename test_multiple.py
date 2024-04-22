
##### TESTING OF ALL EXPERIMENTS FOR PAPER #####

import os
import sys
import subprocess

cifar_label_trainingweight_invloss_expnames = ['CIFAR_label_DAC_invloss_tau_10_seed_1_fixed',
                                               'CIFAR_label_DAC_invloss_tau_10_seed_2_fixed',
                                               'CIFAR_label_DAC_invloss_tau_10_seed_3_fixed',]

cifar_label_trainingweight_l2_expnames = ['CIFAR_label_DAC_l2_tau_30_seed_1_fixed',
                                        'CIFAR_label_DAC_l2_tau_30_seed_2_fixed',
                                        'CIFAR_label_DAC_l2_tau_30_seed_3_fixed',]

cifar_label_trainingweight_cosine_expnames = ['CIFAR_label_DAC_cosine_tau_300_seed_1_fixed',
                                                  'CIFAR_label_DAC_cosine_tau_300_seed_2_fixed',
                                                  'CIFAR_label_DAC_cosine_tau_300_seed_3_fixed',]

cifar_label_trainingweight_origin_expnames = ['CIFAR_label_DAC_cosine_origin_tau_200_seed_1_fixed',
                                                  'CIFAR_label_DAC_cosine_origin_tau_200_seed_2_fixed',
                                                  'CIFAR_label_DAC_cosine_origin_tau_200_seed_3_fixed',]


#-----
cifar_label_priorweight_invloss_expnames = ['CIFAR_label_DAC_priorweight_invloss_tau_5_seed_1_fixed',
                                               'CIFAR_label_DAC_priorweight_invloss_tau_5_seed_2_fixed',
                                               'CIFAR_label_DAC_priorweight_invloss_tau_5_seed_3_fixed',]

cifar_label_priorweight_l2_expnames = ['CIFAR_label_DAC_priorweight_l2_tau_5_seed_1_fixed',
                                        'CIFAR_label_DAC_priorweight_l2_tau_5_seed_2_fixed',
                                        'CIFAR_label_DAC_priorweight_l2_tau_5_seed_3_fixed',]

cifar_label_priorweight_cosine_expnames = ['CIFAR_label_DAC_priorweight_cosine_tau_200_seed_1_fixed',
                                                    'CIFAR_label_DAC_priorweight_cosine_tau_200_seed_2_fixed',
                                                    'CIFAR_label_DAC_priorweight_cosine_tau_200_seed_3_fixed',]

cifar_label_priorweight_origin_expnames = ['CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_1_fixed',
                                                    'CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_2_fixed',
                                                    'CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_3_fixed',]


#-----
cifar_5_clusters_trainingweight_invloss_expnames = ['CIFAR_5_clusters_DAC_invloss_tau_5_seed_1_fixed',
                                               'CIFAR_5_clusters_DAC_invloss_tau_5_seed_2_fixed',
                                               'CIFAR_5_clusters_DAC_invloss_tau_5_seed_3_fixed',]

cifar_5_clusters_trainingweight_l2_expnames = ['CIFAR_5_clusters_DAC_l2_tau_30_seed_1_fixed',
                                        'CIFAR_5_clusters_DAC_l2_tau_30_seed_2_fixed',
                                        'CIFAR_5_clusters_DAC_l2_tau_30_seed_3_fixed',]

cifar_5_clusters_trainingweight_cosine_expnames = ['CIFAR_5_clusters_DAC_cosine_tau_1000_seed_1_fixed',
                                                    'CIFAR_5_clusters_DAC_cosine_tau_1000_seed_2_fixed',
                                                    'CIFAR_5_clusters_DAC_cosine_tau_1000_seed_3_fixed',]

cifar_5_clusters_trainingweight_origin_expnames = ['CIFAR_5_clusters_DAC_cosine_origin_tau_300_seed_1_fixed',
                                                    'CIFAR_5_clusters_DAC_cosine_origin_tau_300_seed_2_fixed',
                                                    'CIFAR_5_clusters_DAC_cosine_origin_tau_300_seed_3_fixed',]



#-----
cifar_5_clusters_priorweight_invloss_expnames = ['CIFAR_5_clusters_DAC_priorweight_invloss_tau_5_seed_1_fixed',
                                               'CIFAR_5_clusters_DAC_priorweight_invloss_tau_5_seed_2_fixed',
                                               'CIFAR_5_clusters_DAC_priorweight_invloss_tau_5_seed_3_fixed',]

cifar_5_clusters_priorweight_l2_expnames = ['CIFAR_5_clusters_DAC_priorweight_l2_tau_10_seed_1_fixed',
                                        'CIFAR_5_clusters_DAC_priorweight_l2_tau_10_seed_2_fixed',
                                        'CIFAR_5_clusters_DAC_priorweight_l2_tau_10_seed_3_fixed',]

cifar_5_clusters_priorweight_cosine_expnames = ['CIFAR_5_clusters_DAC_priorweight_cosine_tau_200_seed_1_fixed',
                                                    'CIFAR_5_clusters_DAC_priorweight_cosine_tau_200_seed_2_fixed',
                                                    'CIFAR_5_clusters_DAC_priorweight_cosine_tau_200_seed_3_fixed',]

cifar_5_clusters_priorweight_origin_expnames = ['CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_200_seed_1_fixed',
                                                    'CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_200_seed_2_fixed',
                                                    'CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_200_seed_3_fixed',]



#-----
fashion_mnist_trainingweight_invloss_expnames = ['fashion_MNIST_DAC_invloss_tau_10_seed_1',
                                                  'fashion_MNIST_DAC_invloss_tau_10_seed_2',
                                                  'fashion_MNIST_DAC_invloss_tau_10_seed_3',]

fashion_mnist_trainingweight_l2_expnames = ['fashion_MNIST_DAC_l2_tau_10_seed_1',
                                                'fashion_MNIST_DAC_l2_tau_10_seed_2',
                                                'fashion_MNIST_DAC_l2_tau_10_seed_3',]

fashion_mnist_trainingweight_cosine_expnames = ['fashion_MNIST_DAC_cosine_tau_2000_seed_1',
                                                    'fashion_MNIST_DAC_cosine_tau_2000_seed_2',
                                                    'fashion_MNIST_DAC_cosine_tau_2000_seed_3',]

fashion_mnist_trainingweight_origin_expnames = ['fashion_MNIST_DAC_cosine_origin_tau_2000_seed_1',
                                                    'fashion_MNIST_DAC_cosine_origin_tau_2000_seed_2',
                                                    'fashion_MNIST_DAC_cosine_origin_tau_2000_seed_3',]


#-----
fashion_mnist_priorweight_invloss_expnames = ['fashion_MNIST_DAC_priorweight_invloss_tau_5_seed_1',
                                               'fashion_MNIST_DAC_priorweight_invloss_tau_5_seed_2',
                                               'fashion_MNIST_DAC_priorweight_invloss_tau_5_seed_3',]

fashion_mnist_priorweight_l2_expnames = ['fashion_MNIST_DAC_priorweight_l2_tau_10_seed_1',
                                        'fashion_MNIST_DAC_priorweight_l2_tau_10_seed_2',
                                        'fashion_MNIST_DAC_priorweight_l2_tau_10_seed_3',]

fashion_mnist_priorweight_cosine_expnames = ['fashion_MNIST_DAC_priorweight_cosine_tau_300_seed_1',
                                                    'fashion_MNIST_DAC_priorweight_cosine_tau_300_seed_2',
                                                    'fashion_MNIST_DAC_priorweight_cosine_tau_300_seed_3',]

fashion_mnist_priorweight_origin_expnames = ['fashion_MNIST_DAC_priorweight_cosine_origin_tau_300_seed_1',
                                                    'fashion_MNIST_DAC_priorweight_cosine_origin_tau_300_seed_2',
                                                    'fashion_MNIST_DAC_priorweight_cosine_origin_tau_300_seed_3',]






##### TESTING #####

all_expnames = [cifar_label_trainingweight_invloss_expnames, 
                fashion_mnist_priorweight_origin_expnames,
                cifar_label_trainingweight_l2_expnames, 
                cifar_label_trainingweight_cosine_expnames, 
                cifar_label_trainingweight_origin_expnames,
                cifar_label_priorweight_invloss_expnames,
                cifar_label_priorweight_l2_expnames,
                cifar_label_priorweight_cosine_expnames,
                cifar_label_priorweight_origin_expnames,
                cifar_5_clusters_trainingweight_invloss_expnames,
                cifar_5_clusters_trainingweight_l2_expnames,
                cifar_5_clusters_trainingweight_cosine_expnames,
                cifar_5_clusters_trainingweight_origin_expnames,
                cifar_5_clusters_priorweight_invloss_expnames,
                cifar_5_clusters_priorweight_l2_expnames,
                cifar_5_clusters_priorweight_cosine_expnames,
                cifar_5_clusters_priorweight_origin_expnames,
                fashion_mnist_trainingweight_invloss_expnames,
                fashion_mnist_trainingweight_l2_expnames,
                fashion_mnist_trainingweight_cosine_expnames,
                fashion_mnist_trainingweight_origin_expnames,
                fashion_mnist_priorweight_invloss_expnames,
                fashion_mnist_priorweight_l2_expnames,
                fashion_mnist_priorweight_cosine_expnames,]


if __name__ == '__main__':

    for expnames in all_expnames:
        for expname in expnames:
            # check if experiment exists
            if not os.path.exists('save/{}'.format(expname)):
                print('Experiment {} does not exist'.format(expname))
                continue
            # check if experiment is finished by checking if metadata has runtime
            try:
                metadata = {}
                with open('save/{}/metadata.txt'.format(expname), 'r') as f:
                    for line in f:
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
            except:
                print('Experiment {}: Metadata not found'.format(expname))
                continue
            # check if runtime is in metadata
            if 'runtime' not in metadata:
                print('Experiment {}: Runtime not found'.format(expname))
                continue

            # split by dataset from now on
            if metadata['dataset'] == 'cifar10':

                if metadata['shift'] == 'label':

                    # check if experiment is already tested
                    if os.path.exists('save/{}/CIFAR_A_within.pkl'.format(expname)) and os.path.exists('save/{}/CIFAR_V_within.pkl'.format(expname)):
                        print('Experiment {} already tested'.format(expname))
                        continue

                    print('Testing experiment: {}'.format(expname))
                    subprocess.run('python3 test_CIFAR.py --experiment {} --quick True'.format(expname), shell=True)
                    print('Experiment {} done'.format(expname))
                    print('')

                elif metadata['shift'] == '5_clusters':

                    # check if experiment is already tested
                    if os.path.exists('save/{}/CIFAR_acc_matrix.pkl'.format(expname)):
                        print('Experiment {} already tested'.format(expname))
                        continue
                        
                    print('Testing experiment: {}'.format(expname))
                    subprocess.run('python3 test_CIFAR.py --experiment {} --quick True'.format(expname), shell=True)
                    print('Experiment {} done'.format(expname))
                    print('')

            elif metadata['dataset'] == 'fashion_mnist':

                # check if experiment is already tested
                if os.path.exists('save/{}/fashion_MNIST_acc_matrix.pkl'.format(expname)):
                    print('Experiment {} already tested'.format(expname))
                    continue

                print('Testing experiment: {}'.format(expname))
                subprocess.run('python3 test_fashion_MNIST.py --experiment {} --quick True'.format(expname), shell=True)
                print('Experiment {} done'.format(expname))
                print('')
