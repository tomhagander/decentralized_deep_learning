
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
hundred_pretrained_invloss_tauopt = 1 # not determined
hundred_pretrained_l2_tauopt = 1 # not determined
hundred_pretrained_cosine_tauopt = 1 # not determined
hundred_pretrained_cosine_origin_tauopt = 1 # not determined
hundred_nonpretrained_invloss_tauopt = 1 # not determined
hundred_nonpretrained_l2_tauopt = 1 # not determined
hundred_nonpretrained_cosine_tauopt = 1 # not determined
hundred_nonpretrained_cosine_origin_tauopt = 1 # not determined

seeds = [1,2,3]
hundred_pretrained_invloss_expnames = ['HUNDRED_pretrained_invloss_tau_{}_seed_{}'.format(hundred_pretrained_invloss_tauopt, seed) for seed in seeds]
hundred_pretrained_l2_expnames = ['HUNDRED_pretrained_l2_tau_{}_seed_{}'.format(hundred_pretrained_l2_tauopt, seed) for seed in seeds]
hundred_pretrained_cosine_expnames = ['HUNDRED_pretrained_cosine_tau_{}_seed_{}'.format(hundred_pretrained_cosine_tauopt, seed) for seed in seeds]
hundred_pretrained_cosine_origin_expnames = ['HUNDRED_pretrained_cosine_origin_tau_{}_seed_{}'.format(hundred_pretrained_cosine_origin_tauopt, seed) for seed in seeds]
hundred_nonpretrained_invloss_expnames = ['HUNDRED_nonpretrained_invloss_tau_{}_seed_{}'.format(hundred_nonpretrained_invloss_tauopt, seed) for seed in seeds]
hundred_nonpretrained_l2_expnames = ['HUNDRED_nonpretrained_l2_tau_{}_seed_{}'.format(hundred_nonpretrained_l2_tauopt, seed) for seed in seeds]
hundred_nonpretrained_cosine_expnames = ['HUNDRED_nonpretrained_cosine_tau_{}_seed_{}'.format(hundred_nonpretrained_cosine_tauopt, seed) for seed in seeds]
hundred_nonpretrained_cosine_origin_expnames = ['HUNDRED_nonpretrained_cosine_origin_tau_{}_seed_{}'.format(hundred_nonpretrained_cosine_origin_tauopt, seed) for seed in seeds]


#### DOUBLE ####
DOUBLE_invloss_expnames = []
DOUBLE_cosine_expnames = []
DOUBLE_origin_expnames = []
DOUBLE_l2_expnames = []

DOUBLE_fedsim_invloss_expnames = []
DOUBLE_fedsim_cosine_expnames = []
DOUBLE_fedsim_origin_expnames = []
DOUBLE_fedsim_l2_expnames = []
for seed in [1,2,3]:
    DOUBLE_invloss_expnames.append('DOUBLE_invloss_seed_{}_tau_1'.format(seed))
    DOUBLE_cosine_expnames.append('DOUBLE_cosine_seed_{}_tau_2000'.format(seed))
    DOUBLE_origin_expnames.append('DOUBLE_origin_seed_{}_tau_2000'.format(seed))
    DOUBLE_l2_expnames.append('DOUBLE_l2_seed_{}_tau_1'.format(seed))

    DOUBLE_fedsim_invloss_expnames.append('DOUBLE_priorweights_invloss_seed_{}_tau_1'.format(seed))
    DOUBLE_fedsim_cosine_expnames.append('DOUBLE_priorweights_cosine_seed_{}_tau_100'.format(seed))
    DOUBLE_fedsim_origin_expnames.append('DOUBLE_priorweights_cosine_origin_seed_{}_tau_300'.format(seed))
    DOUBLE_fedsim_l2_expnames.append('DOUBLE_priorweights_l2_seed_{}_tau_30'.format(seed))

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
                DOUBLE_fedsim_l2_expnames]


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
