
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

fashion_mnist_PANM_invloss_expnames = ['fashion_MNIST_PANM_invloss_seed_1',
                                        'fashion_MNIST_PANM_invloss_seed_2',
                                        'fashion_MNIST_PANM_invloss_seed_3',]

fashiom_mnist_PANM_cosine_expnames = ['fashion_MNIST_PANM_cosine_seed_1',
                                        'fashion_MNIST_PANM_cosine_seed_2',
                                        'fashion_MNIST_PANM_cosine_seed_3',]

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


## Benchmarks
cifar_label_benchmark_no_comm = ['CIFAR_label_no_comm_seed_1_fixed',
                          'CIFAR_label_no_comm_seed_2_fixed',
                          'CIFAR_label_no_comm_seed_3_fixed',]

cifar_label_benchmark_oracle = ['CIFAR_label_oracle_seed_1_fixed',
                            'CIFAR_label_oracle_seed_2_fixed',
                            'CIFAR_label_oracle_seed_3_fixed',]

cifar_label_benchmark_random = ['CIFAR_label_random_seed_1_fixed', 
                            'CIFAR_label_random_seed_2_fixed',
                            'CIFAR_label_random_seed_3_fixed',]

cifar_5_clusters_benchmark_no_comm = ['CIFAR_5_clusters_no_comm_seed_1_fixed', 
                               'CIFAR_5_clusters_no_comm_seed_2_fixed',
                               'CIFAR_5_clusters_no_comm_seed_3_fixed',]

cifar_5_clusters_benchmark_oracle = ['CIFAR_5_clusters_oracle_seed_1_fixed',
                            'CIFAR_5_clusters_oracle_seed_2_fixed',
                            'CIFAR_5_clusters_oracle_seed_3_fixed',]

cifar_5_clusters_benchmark_random = ['CIFAR_5_clusters_random_seed_1_fixed',
                            'CIFAR_5_clusters_random_seed_2_fixed',
                            'CIFAR_5_clusters_random_seed_3_fixed',]

fashion_mnist_benchmark_no_comm = ['fashion_MNIST_no_comm_seed_1', 
                            'fashion_MNIST_no_comm_seed_2',
                            'fashion_MNIST_no_comm_seed_3',]

fashion_mnist_benchmark_oracle = ['fashion_MNIST_oracle_tuning_lr_0.0003_seed_1',
                            'fashion_MNIST_oracle_seed_2_fixed',
                            'fashion_MNIST_oracle_seed_3_fixed',]

fashion_mnist_benchmark_random = ['fashion_MNIST_random_seed_1_fixed', 
                            'fashion_MNIST_random_seed_2_fixed',
                            'fashion_MNIST_random_seed_3_fixed',]

cifar_label_trainingweight_invloss_5epochs_expnames = ['CIFAR_label_DAC_invloss_tau_10_seed_1_5epochs',
                                                        'CIFAR_label_DAC_invloss_tau_10_seed_2_5epochs',
                                                        'CIFAR_label_DAC_invloss_tau_10_seed_3_5epochs',]

cifar_label_trainingweight_l2_5epochs_expnames = ['CIFAR_label_DAC_l2_tau_30_seed_1_5epochs',
                                                  'CIFAR_label_DAC_l2_tau_30_seed_2_5epochs',
                                                  'CIFAR_label_DAC_l2_tau_30_seed_3_5epochs',]

cifar_label_trainingweight_cosine_5epochs_expnames = ['CIFAR_label_DAC_cosine_tau_300_seed_1_5epochs',
                                                      'CIFAR_label_DAC_cosine_tau_300_seed_2_5epochs',
                                                      'CIFAR_label_DAC_cosine_tau_300_seed_3_5epochs',]

cifar_label_trainingweight_origin_5epochs_expnames = ['CIFAR_label_DAC_cosine_origin_tau_200_seed_1_5epochs',
                                                        'CIFAR_label_DAC_cosine_origin_tau_200_seed_2_5epochs',
                                                        'CIFAR_label_DAC_cosine_origin_tau_200_seed_3_5epochs',]

cifar_label_trainingweight_invloss_10epochs_expnames = ['CIFAR_label_DAC_invloss_tau_10_seed_1_10epochs',
                                                        'CIFAR_label_DAC_invloss_tau_10_seed_2_10epochs',
                                                        'CIFAR_label_DAC_invloss_tau_10_seed_3_10epochs',]

cifar_label_trainingweight_l2_10epochs_expnames = ['CIFAR_label_DAC_l2_tau_30_seed_1_10epochs',
                                                    'CIFAR_label_DAC_l2_tau_30_seed_2_10epochs',
                                                    'CIFAR_label_DAC_l2_tau_30_seed_3_10epochs',]

cifar_label_trainingweight_cosine_10epochs_expnames = ['CIFAR_label_DAC_cosine_tau_300_seed_1_10epochs',
                                                        'CIFAR_label_DAC_cosine_tau_300_seed_2_10epochs',
                                                        'CIFAR_label_DAC_cosine_tau_300_seed_3_10epochs',]

cifar_label_trainingweight_origin_10epochs_expnames = ['CIFAR_label_DAC_cosine_origin_tau_200_seed_1_10epochs',
                                                        'CIFAR_label_DAC_cosine_origin_tau_200_seed_2_10epochs',
                                                        'CIFAR_label_DAC_cosine_origin_tau_200_seed_3_10epochs',]

cifar_label_priorweight_invloss_5epochs_expnames = ['CIFAR_label_DAC_priorweight_invloss_tau_5_seed_1_5epochs',
                                                    'CIFAR_label_DAC_priorweight_invloss_tau_5_seed_2_5epochs',
                                                    'CIFAR_label_DAC_priorweight_invloss_tau_5_seed_3_5epochs',]

cifar_label_priorweight_l2_5epochs_expnames = ['CIFAR_label_DAC_priorweight_l2_tau_5_seed_1_5epochs',
                                                'CIFAR_label_DAC_priorweight_l2_tau_5_seed_2_5epochs',
                                                'CIFAR_label_DAC_priorweight_l2_tau_5_seed_3_5epochs',]

cifar_label_priorweight_cosine_5epochs_expnames = ['CIFAR_label_DAC_priorweight_cosine_tau_200_seed_1_5epochs',
                                                    'CIFAR_label_DAC_priorweight_cosine_tau_200_seed_2_5epochs',
                                                    'CIFAR_label_DAC_priorweight_cosine_tau_200_seed_3_5epochs',]

cifar_label_priorweight_origin_5epochs_expnames = ['CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_1_5epochs',
                                                    'CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_2_5epochs',
                                                    'CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_3_5epochs',]

cifar_label_priorweight_invloss_10epochs_expnames = ['CIFAR_label_DAC_priorweight_invloss_tau_5_seed_1_10epochs',
                                                    'CIFAR_label_DAC_priorweight_invloss_tau_5_seed_2_10epochs',
                                                    'CIFAR_label_DAC_priorweight_invloss_tau_5_seed_3_10epochs',]

cifar_label_priorweight_l2_10epochs_expnames = ['CIFAR_label_DAC_priorweight_l2_tau_5_seed_1_10epochs',
                                                'CIFAR_label_DAC_priorweight_l2_tau_5_seed_2_10epochs',
                                                'CIFAR_label_DAC_priorweight_l2_tau_5_seed_3_10epochs',]

cifar_label_priorweight_cosine_10epochs_expnames = ['CIFAR_label_DAC_priorweight_cosine_tau_200_seed_1_10epochs',
                                                    'CIFAR_label_DAC_priorweight_cosine_tau_200_seed_2_10epochs',
                                                    'CIFAR_label_DAC_priorweight_cosine_tau_200_seed_3_10epochs',]

cifar_label_PANM_invloss_expnames = ['CIFAR_label_PANM_invloss_seed_1',
                                        'CIFAR_label_PANM_invloss_seed_2',
                                        'CIFAR_label_PANM_invloss_seed_3',]

cifar_label_PANM_cosine_expnames = ['CIFAR_label_PANM_cosine_seed_1',
                                        'CIFAR_label_PANM_cosine_seed_2',
                                        'CIFAR_label_PANM_cosine_seed_3',]




#----
toy_benchmark_oracle = ['TOY_oracle_seed_1',
                        'TOY_oracle_seed_2',
                        'TOY_oracle_seed_3',]

toy_benchmark_random = ['TOY_random_seed_1',
                        'TOY_random_seed_2',
                        'TOY_random_seed_3',]

toy_benchmark_no_comm = ['TOY_no_comm_seed_1',
                        'TOY_no_comm_seed_2',
                        'TOY_no_comm_seed_3',]

# taus not set from here on out. Might also want more runs here

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
lr = 0.003
toy_trainingweight_invloss_tauopt = 10000 # not determined
toy_trainingweight_l2_tauopt = 19
toy_trainingweight_cosine_tauopt = 140
toy_trainingweight_cosine_origin_tauopt = 140
toy_priorweight_invloss_tauopt = 5000 # not determined
toy_priorweight_l2_tauopt = 19
toy_priorweight_cosine_tauopt = 140
toy_priorweight_cosine_origin_tauopt = 140

toy_trainingweight_invloss_expnames = []
toy_trainingweight_l2_expnames = []
toy_trainingweight_cosine_expnames = []
toy_trainingweight_origin_expnames = []
toy_priorweight_invloss_expnames = []
toy_priorweight_l2_expnames = []
toy_priorweight_cosine_expnames = []
toy_priorweight_origin_expnames = []


for seed in seeds:
    toy_trainingweight_invloss_expnames.append('TOY_DAC_invloss_tau_{}_seed_{}'.format(toy_trainingweight_invloss_tauopt, seed))
    toy_trainingweight_l2_expnames.append('TOY_DAC_l2_tau_{}_seed_{}'.format(toy_trainingweight_l2_tauopt, seed))
    toy_trainingweight_cosine_expnames.append('TOY_DAC_cosine_tau_{}_seed_{}'.format(toy_trainingweight_cosine_tauopt, seed))
    toy_trainingweight_origin_expnames.append('TOY_DAC_cosine_origin_tau_{}_seed_{}'.format(toy_trainingweight_cosine_origin_tauopt, seed))
    toy_priorweight_invloss_expnames.append('TOY_DAC_priorweight_invloss_tau_{}_seed_{}'.format(toy_priorweight_invloss_tauopt, seed))
    toy_priorweight_l2_expnames.append('TOY_DAC_priorweight_l2_tau_{}_seed_{}'.format(toy_priorweight_l2_tauopt, seed))
    toy_priorweight_cosine_expnames.append('TOY_DAC_priorweight_cosine_tau_{}_seed_{}'.format(toy_priorweight_cosine_tauopt, seed))
    toy_priorweight_origin_expnames.append('TOY_DAC_priorweight_cosine_origin_tau_{}_seed_{}'.format(toy_priorweight_cosine_origin_tauopt, seed))


#### Toy minmax ####
toy_trainingweight_invloss_tauopt = 10
toy_trainingweight_l2_tauopt = 10
toy_trainingweight_cosine_tauopt = 100
toy_trainingweight_cosine_origin_tauopt = 100

seeds = [1, 3, 4, 7]

toy_minmax_invloss_expnames = []
toy_minmax_l2_expnames = []
toy_minmax_cosine_expnames = []
toy_minmax_cosine_origin_expnames = []

for seed in seeds:
    toy_minmax_invloss_expnames.append('TOY_DAC_minmax_invloss_tau_{}_seed_{}'.format(toy_trainingweight_invloss_tauopt, seed))
    toy_minmax_l2_expnames.append('TOY_DAC_minmax_l2_tau_{}_seed_{}'.format(toy_trainingweight_l2_tauopt, seed))
    toy_minmax_cosine_expnames.append('TOY_DAC_minmax_cosine_tau_{}_seed_{}'.format(toy_trainingweight_cosine_tauopt, seed))
    toy_minmax_cosine_origin_expnames.append('TOY_DAC_minmax_cosine_origin_tau_{}_seed_{}'.format(toy_trainingweight_cosine_origin_tauopt, seed))

# priorweight
toy_priorweight_invloss_tauopt = 10 
toy_priorweight_l2_tauopt = 10
toy_priorweight_cosine_tauopt = 100
toy_priorweight_cosine_origin_tauopt = 100

toy_minmax_priorweight_invloss_expnames = []
toy_minmax_priorweight_l2_expnames = []
toy_minmax_priorweight_cosine_expnames = []
toy_minmax_priorweight_cosine_origin_expnames = []

for seed in seeds:
    toy_minmax_priorweight_invloss_expnames.append('TOY_DAC_minmax_priorweight_invloss_tau_{}_seed_{}'.format(toy_priorweight_invloss_tauopt, seed))
    toy_minmax_priorweight_l2_expnames.append('TOY_DAC_minmax_priorweight_l2_tau_{}_seed_{}'.format(toy_priorweight_l2_tauopt, seed))
    toy_minmax_priorweight_cosine_expnames.append('TOY_DAC_minmax_priorweight_cosine_tau_{}_seed_{}'.format(toy_priorweight_cosine_tauopt, seed))
    toy_minmax_priorweight_cosine_origin_expnames.append('TOY_DAC_minmax_priorweight_cosine_origin_tau_{}_seed_{}'.format(toy_priorweight_cosine_origin_tauopt, seed))


##### HUNDRED #####
# benchmarks
hundred_pretrained_no_comm_expnames = ['HUNDRED_pretrained_no_comm_seed_1',
                                'HUNDRED_pretrained_no_comm_seed_2',
                                'HUNDRED_pretrained_no_comm_seed_3',]

hundred_pretrained_oracle_expnames = ['HUNDRED_pretrained_oracle_seed_1',
                                'HUNDRED_pretrained_oracle_seed_2',
                                'HUNDRED_pretrained_oracle_seed_3',]

hundred_pretrained_random_expnames = ['HUNDRED_pretrained_random_seed_1',
                                'HUNDRED_pretrained_random_seed_2',
                                'HUNDRED_pretrained_random_seed_3',]

hundred_nonpretrained_no_comm_expnames = ['HUNDRED_nonpretrained_no_comm_seed_1',
                                'HUNDRED_nonpretrained_no_comm_seed_2',
                                'HUNDRED_nonpretrained_no_comm_seed_3',]

hundred_nonpretrained_oracle_expnames = ['HUNDRED_nonpretrained_oracle_seed_1',
                                'HUNDRED_nonpretrained_oracle_seed_2',
                                'HUNDRED_nonpretrained_oracle_seed_3',]

hundred_nonpretrained_random_expnames = ['HUNDRED_nonpretrained_random_seed_1',
                                'HUNDRED_nonpretrained_random_seed_2',
                                'HUNDRED_nonpretrained_random_seed_3',]

# runs
hundred_pretrained_invloss_tauopt = 1
hundred_pretrained_l2_tauopt = 30
hundred_pretrained_cosine_tauopt = 5
hundred_pretrained_cosine_origin_tauopt = 30
hundred_nonpretrained_invloss_tauopt = 5
hundred_nonpretrained_l2_tauopt = 5
hundred_nonpretrained_cosine_tauopt = 5
hundred_nonpretrained_cosine_origin_tauopt = 30

seeds = [2,3]
hundred_pretrained_invloss_expnames = ['HUNDRED_pretrained_invloss_tau_{}_seed_{}'.format(hundred_pretrained_invloss_tauopt, seed) for seed in seeds]
hundred_pretrained_invloss_expnames.append('HUNDRED_pretrained_invloss_tau_1.0')
hundred_pretrained_l2_expnames = ['HUNDRED_pretrained_l2_tau_{}_seed_{}'.format(hundred_pretrained_l2_tauopt, seed) for seed in seeds]
hundred_pretrained_l2_expnames.append('HUNDRED_pretrained_l2_tau_28.231080866430865')
hundred_pretrained_cosine_expnames = ['HUNDRED_pretrained_cosine_tau_{}_seed_{}'.format(hundred_pretrained_cosine_tauopt, seed) for seed in seeds]
hundred_pretrained_cosine_expnames.append('HUNDRED_pretrained_cosine_tau_5.0')
hundred_pretrained_cosine_origin_expnames = ['HUNDRED_pretrained_cosine_origin_tau_{}_seed_{}'.format(hundred_pretrained_cosine_origin_tauopt, seed) for seed in seeds]
hundred_pretrained_cosine_origin_expnames.append('HUNDRED_pretrained_cosine_origin_tau_31.0723250595386')
hundred_nonpretrained_invloss_expnames = ['HUNDRED_nonpretrained_invloss_tau_{}_seed_{}'.format(hundred_nonpretrained_invloss_tauopt, seed) for seed in seeds]
hundred_nonpretrained_invloss_expnames.append('HUNDRED_nonpretrained_invloss_tau_5.313292845913056')
hundred_nonpretrained_l2_expnames = ['HUNDRED_nonpretrained_l2_tau_{}_seed_{}'.format(hundred_nonpretrained_l2_tauopt, seed) for seed in seeds]
hundred_nonpretrained_l2_expnames.append('HUNDRED_nonpretrained_l2_tau_5.313292845913056')
hundred_nonpretrained_cosine_expnames = ['HUNDRED_nonpretrained_cosine_tau_{}_seed_{}'.format(hundred_nonpretrained_cosine_tauopt, seed) for seed in seeds]
hundred_nonpretrained_cosine_expnames.append('HUNDRED_nonpretrained_cosine_tau_5.0')
hundred_nonpretrained_cosine_origin_expnames = ['HUNDRED_nonpretrained_cosine_origin_tau_{}_seed_{}'.format(hundred_nonpretrained_cosine_origin_tauopt, seed) for seed in seeds]
hundred_nonpretrained_cosine_origin_expnames.append('HUNDRED_nonpretrained_cosine_origin_tau_31.0723250595386')


#### DOUBLE ####
DOUBLE_invloss_expnames = []
DOUBLE_cosine_expnames = []
DOUBLE_origin_expnames = []
DOUBLE_l2_expnames = []

DOUBLE_fedsim_invloss_expnames = []
DOUBLE_fedsim_cosine_expnames = []
DOUBLE_fedsim_origin_expnames = []
DOUBLE_fedsim_l2_expnames = []

DOUBLE_oracle_expnames = []
DOUBLE_random_expnames = []
DOUBLE_no_comm_expnames = []
for seed in [1,2,3]:
    DOUBLE_invloss_expnames.append('DOUBLE_invloss_seed_{}_tau_1'.format(seed))
    DOUBLE_cosine_expnames.append('DOUBLE_cosine_seed_{}_tau_2000'.format(seed))
    DOUBLE_origin_expnames.append('DOUBLE_cosine_origin_seed_{}_tau_2000'.format(seed))
    DOUBLE_l2_expnames.append('DOUBLE_l2_seed_{}_tau_30'.format(seed))

    DOUBLE_fedsim_invloss_expnames.append('DOUBLE_priorweight_invloss_seed_{}_tau_1'.format(seed))
    DOUBLE_fedsim_cosine_expnames.append('DOUBLE_priorweight_cosine_seed_{}_tau_100'.format(seed))
    DOUBLE_fedsim_origin_expnames.append('DOUBLE_priorweight_cosine_origin_seed_{}_tau_300'.format(seed))
    DOUBLE_fedsim_l2_expnames.append('DOUBLE_priorweight_l2_seed_{}_tau_30'.format(seed))

    DOUBLE_oracle_expnames.append('DOUBLE_oracle_seed_{}'.format(seed))
    DOUBLE_random_expnames.append('DOUBLE_random_seed_{}'.format(seed))
    DOUBLE_no_comm_expnames.append('DOUBLE_no_comm_seed_{}'.format(seed))



#### TOY Trainset size experiments ####
# seeds one to 15
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
trainset_sizes = [10, 100, 200]

toy_trainingweight_invloss_tauopt = 10000
toy_trainingweight_l2_tauopt = 19
toy_trainingweight_cosine_tauopt = 140
toy_trainingweight_cosine_origin_tauopt = 140
toy_priorweight_invloss_tauopt = 5000
toy_priorweight_l2_tauopt = 19
toy_priorweight_cosine_tauopt = 140
toy_priorweight_cosine_origin_tauopt = 140

lr = 0.003
lr_invloss_l2 = 0.008

toy_ts50_trainingweight_invloss_expnames = []
toy_ts50_trainingweight_l2_expnames = []
toy_ts50_trainingweight_cosine_expnames = []
toy_ts50_trainingweight_origin_expnames = []
toy_ts50_priorweight_invloss_expnames = []
toy_ts50_priorweight_l2_expnames = []
toy_ts50_priorweight_cosine_expnames = []
toy_ts50_priorweight_origin_expnames = []
toy_ts100_trainingweight_invloss_expnames = []
toy_ts100_trainingweight_l2_expnames = []
toy_ts100_trainingweight_cosine_expnames = []
toy_ts100_trainingweight_origin_expnames = []
toy_ts100_priorweight_invloss_expnames = []
toy_ts100_priorweight_l2_expnames = []
toy_ts100_priorweight_cosine_expnames = []
toy_ts100_priorweight_origin_expnames = []
toy_ts200_trainingweight_invloss_expnames = []
toy_ts200_trainingweight_l2_expnames = []
toy_ts200_trainingweight_cosine_expnames = []
toy_ts200_trainingweight_origin_expnames = []
toy_ts200_priorweight_invloss_expnames = []
toy_ts200_priorweight_l2_expnames = []
toy_ts200_priorweight_cosine_expnames = []
toy_ts200_priorweight_origin_expnames = []


for seed in seeds:
    # invloss trainingweight ts 50
    expname = 'TOY_invloss_trainingweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_trainingweight_invloss_expnames.append(expname)
    # l2 trainingweight ts 50
    expname = 'TOY_l2_trainingweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_trainingweight_l2_expnames.append(expname)
    # cosine trainingweight ts 50
    expname = 'TOY_cosine_trainingweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_trainingweight_cosine_expnames.append(expname)
    # cosine origin trainingweight ts 50
    expname = 'TOY_cosine_origin_trainingweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_trainingweight_origin_expnames.append(expname)
    # invloss priorweight ts 50
    expname = 'TOY_invloss_priorweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_priorweight_invloss_expnames.append(expname)
    # l2 priorweight ts 50
    expname = 'TOY_l2_priorweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_priorweight_l2_expnames.append(expname)
    # cosine priorweight ts 50
    expname = 'TOY_cosine_priorweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_priorweight_cosine_expnames.append(expname)
    # cosine origin priorweight ts 50
    expname = 'TOY_cosine_origin_priorweight_trainset_size_10_seed_{}'.format(seed)
    toy_ts50_priorweight_origin_expnames.append(expname)
    # invloss trainingweight ts 100
    expname = 'TOY_invloss_trainingweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_trainingweight_invloss_expnames.append(expname)
    # l2 trainingweight ts 100
    expname = 'TOY_l2_trainingweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_trainingweight_l2_expnames.append(expname)
    # cosine trainingweight ts 100
    expname = 'TOY_cosine_trainingweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_trainingweight_cosine_expnames.append(expname)
    # cosine origin trainingweight ts 100
    expname = 'TOY_cosine_origin_trainingweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_trainingweight_origin_expnames.append(expname)
    # invloss priorweight ts 100
    expname = 'TOY_invloss_priorweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_priorweight_invloss_expnames.append(expname)
    # l2 priorweight ts 100
    expname = 'TOY_l2_priorweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_priorweight_l2_expnames.append(expname)
    # cosine priorweight ts 100
    expname = 'TOY_cosine_priorweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_priorweight_cosine_expnames.append(expname)
    # cosine origin priorweight ts 100
    expname = 'TOY_cosine_origin_priorweight_trainset_size_100_seed_{}'.format(seed)
    toy_ts100_priorweight_origin_expnames.append(expname)
    # invloss trainingweight ts 200
    expname = 'TOY_invloss_trainingweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_trainingweight_invloss_expnames.append(expname)
    # l2 trainingweight ts 200
    expname = 'TOY_l2_trainingweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_trainingweight_l2_expnames.append(expname)
    # cosine trainingweight ts 200
    expname = 'TOY_cosine_trainingweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_trainingweight_cosine_expnames.append(expname)
    # cosine origin trainingweight ts 200
    expname = 'TOY_cosine_origin_trainingweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_trainingweight_origin_expnames.append(expname)
    # invloss priorweight ts 200
    expname = 'TOY_invloss_priorweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_priorweight_invloss_expnames.append(expname)
    # l2 priorweight ts 200
    expname = 'TOY_l2_priorweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_priorweight_l2_expnames.append(expname)
    # cosine priorweight ts 200
    expname = 'TOY_cosine_priorweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_priorweight_cosine_expnames.append(expname)
    # cosine origin priorweight ts 200
    expname = 'TOY_cosine_origin_priorweight_trainset_size_200_seed_{}'.format(seed)
    toy_ts200_priorweight_origin_expnames.append(expname)



##### DOUBLE MLP #####

DOUBLE_MLP_invloss_expnames = []
DOUBLE_MLP_cosine_expnames = []
DOUBLE_MLP_origin_expnames = []
DOUBLE_MLP_l2_expnames = []
DOUBLE_MLP_priorweight_invloss_expnames = []
DOUBLE_MLP_priorweight_cosine_expnames = []
DOUBLE_MLP_priorweight_origin_expnames = []
DOUBLE_MLP_priorweight_l2_expnames = []
DOUBLE_MLP_oracle_expnames = []
DOUBLE_MLP_random_expnames = []
DOUBLE_MLP_no_comm_expnames = []

trainingweight_invloss_tauopt = 5
trainingweight_l2_tauopt = 10
trainingweight_cosine_tauopt = 3000 
trainingweight_cosine_origin_tauopt = 140 # not set, the only one left
priorweight_invloss_tauopt = 1 
priorweight_l2_tauopt = 5 
priorweight_cosine_tauopt = 300 
priorweight_cosine_origin_tauopt = 5

# add optimal run from tuning
DOUBLE_MLP_invloss_expnames.append('DOUBLE_MLP_invloss_trainingweight_tau_{}'.format(trainingweight_invloss_tauopt))
DOUBLE_MLP_l2_expnames.append('DOUBLE_MLP_l2_trainingweight_tau_{}'.format(trainingweight_l2_tauopt))
DOUBLE_MLP_cosine_expnames.append('DOUBLE_MLP_cosine_trainingweight_tau_{}'.format(trainingweight_cosine_tauopt))
DOUBLE_MLP_origin_expnames.append('DOUBLE_MLP_cosine_origin_trainingweight_tau_{}'.format(trainingweight_cosine_origin_tauopt))
DOUBLE_MLP_priorweight_invloss_expnames.append('DOUBLE_MLP_priorweight_invloss_tau_{}'.format(priorweight_invloss_tauopt))
DOUBLE_MLP_priorweight_l2_expnames.append('DOUBLE_MLP_priorweight_l2_tau_{}'.format(priorweight_l2_tauopt))
DOUBLE_MLP_priorweight_cosine_expnames.append('DOUBLE_MLP_priorweight_cosine_tau_{}'.format(priorweight_cosine_tauopt))
DOUBLE_MLP_priorweight_origin_expnames.append('DOUBLE_MLP_priorweight_cosine_origin_tau_{}'.format(priorweight_cosine_origin_tauopt))

# add reproduction runs
for seed in [2,3]:
    expname = 'DOUBLE_MLP_invloss_trainingweight_tau_{}_seed_{}'.format(trainingweight_invloss_tauopt, seed)
    DOUBLE_MLP_invloss_expnames.append(expname)

    expname = 'DOUBLE_MLP_l2_trainingweight_tau_{}_seed_{}'.format(trainingweight_l2_tauopt, seed)
    DOUBLE_MLP_l2_expnames.append(expname)

    expname = 'DOUBLE_MLP_cosine_trainingweight_tau_{}_seed_{}'.format(trainingweight_cosine_tauopt, seed)
    DOUBLE_MLP_cosine_expnames.append(expname)

    expname = 'DOUBLE_MLP_cosine_origin_trainingweight_tau_{}_seed_{}'.format(trainingweight_cosine_origin_tauopt, seed)
    DOUBLE_MLP_origin_expnames.append(expname)

    expname = 'DOUBLE_MLP_priorweight_invloss_tau_{}_seed_{}'.format(priorweight_invloss_tauopt, seed)
    DOUBLE_MLP_priorweight_invloss_expnames.append(expname)

    expname = 'DOUBLE_MLP_priorweight_l2_tau_{}_seed_{}'.format(priorweight_l2_tauopt, seed)
    DOUBLE_MLP_priorweight_l2_expnames.append(expname)

    expname = 'DOUBLE_MLP_priorweight_cosine_tau_{}_seed_{}'.format(priorweight_cosine_tauopt, seed)
    DOUBLE_MLP_priorweight_cosine_expnames.append(expname)

    expname = 'DOUBLE_MLP_priorweight_cosine_origin_tau_{}_seed_{}'.format(priorweight_cosine_origin_tauopt, seed)
    DOUBLE_MLP_priorweight_origin_expnames.append(expname)

# benchmarks
DOUBLE_MLP_oracle_expnames = ['DOUBLE_MLP_oracle_seed_1',
                            'DOUBLE_MLP_oracle_seed_2',
                            'DOUBLE_MLP_oracle_seed_3',]

DOUBLE_MLP_random_expnames = ['DOUBLE_MLP_random_seed_1',
                            'DOUBLE_MLP_random_seed_2',
                            'DOUBLE_MLP_random_seed_3',]

DOUBLE_MLP_no_comm_expnames = ['DOUBLE_MLP_no_comm_seed_1',
                            'DOUBLE_MLP_no_comm_seed_2',
                            'DOUBLE_MLP_no_comm_seed_3',]



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
                fashion_mnist_priorweight_cosine_expnames,
                fashion_mnist_PANM_invloss_expnames,
                fashiom_mnist_PANM_cosine_expnames,
                cifar_label_benchmark_no_comm,
                cifar_label_benchmark_oracle,
                cifar_label_benchmark_random,
                cifar_5_clusters_benchmark_no_comm,
                cifar_5_clusters_benchmark_oracle,
                cifar_5_clusters_benchmark_random,
                fashion_mnist_benchmark_no_comm,
                fashion_mnist_benchmark_oracle,
                fashion_mnist_benchmark_random,
                cifar_label_trainingweight_invloss_5epochs_expnames,
                cifar_label_trainingweight_l2_5epochs_expnames,
                cifar_label_trainingweight_cosine_5epochs_expnames,
                cifar_label_trainingweight_origin_5epochs_expnames,
                cifar_label_trainingweight_invloss_10epochs_expnames,
                cifar_label_trainingweight_l2_10epochs_expnames,
                cifar_label_trainingweight_cosine_10epochs_expnames,
                cifar_label_trainingweight_origin_10epochs_expnames,
                cifar_label_priorweight_invloss_5epochs_expnames,
                cifar_label_priorweight_l2_5epochs_expnames,
                cifar_label_priorweight_cosine_5epochs_expnames,
                cifar_label_priorweight_origin_5epochs_expnames,
                cifar_label_priorweight_invloss_10epochs_expnames,
                cifar_label_priorweight_l2_10epochs_expnames,
                cifar_label_priorweight_cosine_10epochs_expnames,
                cifar_label_PANM_invloss_expnames,
                cifar_label_PANM_cosine_expnames,
                toy_benchmark_oracle,
                toy_benchmark_random,
                toy_benchmark_no_comm,
                toy_trainingweight_invloss_expnames,
                toy_trainingweight_l2_expnames,
                toy_trainingweight_cosine_expnames,
                toy_trainingweight_origin_expnames,
                toy_priorweight_invloss_expnames,
                toy_priorweight_l2_expnames,
                toy_priorweight_cosine_expnames,
                toy_priorweight_origin_expnames,
                toy_minmax_invloss_expnames,
                toy_minmax_l2_expnames,
                toy_minmax_cosine_expnames,
                toy_minmax_cosine_origin_expnames,
                toy_minmax_priorweight_invloss_expnames,
                toy_minmax_priorweight_l2_expnames,
                toy_minmax_priorweight_cosine_expnames,
                toy_minmax_priorweight_cosine_origin_expnames,
                hundred_pretrained_no_comm_expnames,
                hundred_pretrained_oracle_expnames,
                hundred_pretrained_random_expnames,
                hundred_nonpretrained_no_comm_expnames,
                hundred_nonpretrained_oracle_expnames,
                hundred_nonpretrained_random_expnames,
                hundred_pretrained_invloss_expnames,
                hundred_pretrained_l2_expnames,
                hundred_pretrained_cosine_expnames,
                hundred_pretrained_cosine_origin_expnames,
                hundred_nonpretrained_invloss_expnames,
                hundred_nonpretrained_l2_expnames,
                hundred_nonpretrained_cosine_expnames,
                hundred_nonpretrained_cosine_origin_expnames, 
                DOUBLE_invloss_expnames,
                DOUBLE_cosine_expnames,
                DOUBLE_origin_expnames,
                DOUBLE_l2_expnames,
                DOUBLE_fedsim_invloss_expnames,
                DOUBLE_fedsim_cosine_expnames,
                DOUBLE_fedsim_origin_expnames,
                DOUBLE_fedsim_l2_expnames,
                DOUBLE_oracle_expnames,
                DOUBLE_random_expnames,
                DOUBLE_no_comm_expnames,
                toy_ts50_trainingweight_invloss_expnames,
                toy_ts50_trainingweight_l2_expnames,
                toy_ts50_trainingweight_cosine_expnames,
                toy_ts50_trainingweight_origin_expnames,
                toy_ts50_priorweight_invloss_expnames,
                toy_ts50_priorweight_l2_expnames,
                toy_ts50_priorweight_cosine_expnames,
                toy_ts50_priorweight_origin_expnames,
                toy_ts100_trainingweight_invloss_expnames,
                toy_ts100_trainingweight_l2_expnames,
                toy_ts100_trainingweight_cosine_expnames,
                toy_ts100_trainingweight_origin_expnames,
                toy_ts100_priorweight_invloss_expnames,
                toy_ts100_priorweight_l2_expnames,
                toy_ts100_priorweight_cosine_expnames,
                toy_ts100_priorweight_origin_expnames,
                toy_ts200_trainingweight_invloss_expnames,
                toy_ts200_trainingweight_l2_expnames,
                toy_ts200_trainingweight_cosine_expnames,
                toy_ts200_trainingweight_origin_expnames,
                toy_ts200_priorweight_invloss_expnames,
                toy_ts200_priorweight_l2_expnames,
                toy_ts200_priorweight_cosine_expnames,
                toy_ts200_priorweight_origin_expnames,
                DOUBLE_no_comm_expnames,
                DOUBLE_MLP_invloss_expnames,
                DOUBLE_MLP_cosine_expnames,
                DOUBLE_MLP_origin_expnames,
                DOUBLE_MLP_l2_expnames,
                DOUBLE_MLP_priorweight_invloss_expnames,
                DOUBLE_MLP_priorweight_cosine_expnames,
                DOUBLE_MLP_priorweight_origin_expnames,
                DOUBLE_MLP_priorweight_l2_expnames,
                DOUBLE_MLP_oracle_expnames,
                DOUBLE_MLP_random_expnames,
                DOUBLE_MLP_no_comm_expnames,]


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
            if metadata['runtime'] == 42:
                print('no_clients_flag')
            else:
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

                elif metadata['dataset'] == 'toy_problem':

                    # check if experiment is already tested
                    if os.path.exists('save/{}/toy_test_losses.pkl'.format(expname)):
                        print('Experiment {} already tested'.format(expname))
                        continue
                
                    print('Testing experiment: {}'.format(expname))
                    subprocess.run('python3 test_toy.py --experiment {} --quick True'.format(expname), shell=True)
                    print('Experiment {} done'.format(expname))
                    print('')

                elif metadata['dataset'] == 'cifar100':
                    print('CIFAR100 already tested')

                elif metadata['dataset'] == 'double':
                    # check if experiment is already tested
                    if os.path.exists('save/{}/double_MNIST_acc_matrix.pkl'.format(expname)):
                        print('Experiment {} already tested'.format(expname))
                        continue

                    print('Testing experiment: {}'.format(expname))
                    subprocess.run('python3 test_double_MNIST.py --experiment {} --quick True'.format(expname), shell=True)
                    print('Experiment {} done'.format(expname))
                    print('')
