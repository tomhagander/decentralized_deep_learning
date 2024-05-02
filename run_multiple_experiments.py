import subprocess

commands = []

lr = 0.001  
# lr = 0.01

'''
# PANM_swap4 with cifarratio 0.25
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange oracle --delusion 0 --experiment_name RECREATE_ORACLE_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with random
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange oracle --delusion -1 --experiment_name RECREATE_RANDOM_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with DAC, variable tau, tau 30, inverse_training loss
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --similarity_metric inverse_training_loss --experiment_name RECREATE_DAC_INV_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with DAC, variable tau, tau 30, cosine similarity, alpha 0.5

'''
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange DAC --tau 30 --similarity_metric cosine_similarity --cosine_alpha 0.5 --experiment_name RECREATE_DAC_COS_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with PANM, T1 100, NAEM frequency 2, inverse_training_loss
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange PANM --T1 100 --NAEM_frequency 2 --similarity_metric inverse_training_loss --experiment_name RECREATE_PANM_INV_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with PANM, T1 100, NAEM frequency 2, cosine similarity, alpha 0.5
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange PANM --T1 100 --NAEM_frequency 2 --similarity_metric cosine_similarity --cosine_alpha 0.5 --experiment_name RECREATE_PANM_COS_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))

#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange DAC --prior_update_rule softmax-fixed-entropy --similarity_metric inverse_training_loss --measure_all_similarities True --experiment_name CIFAR_inverse_loss_fixed_entropy_learning_rate_{} '.format(lr, lr))

# for tau in [30,10,1]:
    # commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange DAC --tau {} --prior_update_rule softmax --similarity_metric inverse_training_loss --measure_all_similarities True --experiment_name CIFAR_inverse_loss_tau_{}_learning_rate_{} '.format(lr, tau, tau, lr))

# lr = 0.005
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange oracle --similarity_metric inverse_training_loss --measure_all_similarities False --experiment_name CIFAR_oracle_learning_rate_{} '.format(lr, lr))
# lr = 0.0005
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange oracle --similarity_metric inverse_training_loss --measure_all_similarities False --experiment_name CIFAR_oracle_learning_rate_{} '.format(lr, lr))
# lr = 0.0001
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange oracle --similarity_metric inverse_training_loss --measure_all_similarities False --experiment_name CIFAR_oracle_learning_rate_{} '.format(lr, lr))

# for tau in [30]:
#     commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange DAC --tau {} --prior_update_rule softmax --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name CIFAR_cosine_tau_{}_learning_rate_{} '.format(lr, tau, tau, lr))

#no comm
#lr = 0.0005
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange no_exchange --similarity_metric inverse_training_loss --measure_all_similarities True --experiment_name CIFAR_inverse_loss_no_comm_learning_rate_{} '.format(lr, lr))
#lr = 0.0001
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange no_exchange --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name CIFAR_cosine_no_comm_learning_rate_{} '.format(lr, lr))

# random
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange oracle --delusion -1 --similarity_metric inverse_training_loss --measure_all_similarities False --experiment_name CIFAR_random_learning_rate_{} '.format(lr, lr))
#for tau in [10,1]:
#    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange DAC --tau {} --prior_update_rule softmax --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name CIFAR_cosine_tau_{}_learning_rate_{} '.format(lr, tau, tau, lr))


########## BIG SETUP ############
# six runs in parallel, 4 5hr runs, or 2 10 hr runs (maybe 3 or 4 if eric solves measure all sims)

#lr = 0.001
#no_comm_lr = 0.0001

'''

# setup 1
# no comm with measuring invloss and cosine. One run with cosine already done
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_invloss_no_comm_seed_1 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(no_comm_lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_invloss_no_comm_seed_1')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_invloss_no_comm_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(no_comm_lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_invloss_no_comm_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_cosine_no_comm_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(no_comm_lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_cosine_no_comm_seed_2')


# setup 2
# oracle and random
# oracle already has one with correct lr, random maybe
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax-variable-tau --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange oracle --experiment_name CIFAR_random_seed_2 --delusion -1.0 --CIFAR_ratio 0.4 --measure_all_similarities False'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_random_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax-variable-tau --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange oracle --experiment_name CIFAR_oracle_seed_2 --delusion 0 --CIFAR_ratio 0.4 --measure_all_similarities False'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_oracle_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax-variable-tau --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange oracle --experiment_name CIFAR_random_seed_3 --delusion -1.0 --CIFAR_ratio 0.4 --measure_all_similarities False'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_random_seed_3')

# setup 3
# DAC with tau = 30, invloss and cosine. Also one alldata
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_DAC_invloss_tau_30_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_invloss_tau_30_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_DAC_cosine_tau_30_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_cosine_tau_30_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 2 --n_data_train 10000 --n_data_val 5000 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_invloss_alldata_seed_1 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_invloss_alldata_seed_1')

# setup 4
# DAC with tau = 1, invloss and cosine
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_DAC_invloss_tau_1_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_invloss_tau_1_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_DAC_cosine_tau_1_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_cosine_tau_1_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 2 --n_data_train 10000 --n_data_val 5000 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_invloss_alldata_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_invloss_alldata_seed_2')


# setup 5
# DAC with tau = 10, invloss and cosine
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_DAC_invloss_tau_10_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_invloss_tau_10_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_DAC_cosine_tau_10_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_cosine_tau_10_seed_2')

# setup 6
# DAC with fixed entropy tau, invloss and cosine
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax-fixed-entropy --similarity_metric inverse_training_loss --cosine_alpha 0 --client_information_exchange DAC --experiment_name CIFAR_DAC_invloss_tau_FE_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_invloss_tau_FE_seed_2')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax-fixed-entropy --similarity_metric cosine_similarity --cosine_alpha 0 --client_information_exchange DAC --experiment_name CIFAR_DAC_cosine_tau_FE_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_DAC_cosine_tau_FE_seed_2')


# setup 7
# testing of existing runs, should be fairly quick
commands.append('python3 test_CIFAR.py --experiment CIFAR_inverse_loss_fixed_entropy_learning_rate_0.001')
commands.append('python3 test_CIFAR.py --experiment CIFAR_inverse_loss_tau_1_learning_rate_0.001')
commands.append('python3 test_CIFAR.py --experiment CIFAR_inverse_loss_tau_10_learning_rate_0.001')
commands.append('python3 test_CIFAR.py --experiment CIFAR_inverse_loss_tau_30_learning_rate_0.001')
commands.append('python3 test_CIFAR.py --experiment CIFAR_inverse_loss_no_comm_learning_rate_0.0001')
commands.append('python3 test_CIFAR.py --experiment CIFAR_oracle_learning_rate_0.001')
commands.append('python3 test_CIFAR.py --experiment CIFAR_random_learning_rate_0.001')

'''


##### MERGATRON WEEKEND #####

lr = 0.001
no_comm_lr = 0.0001
seeds = [1, 2, 3]

'''

### core 1
# tau 1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_1_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_l2_tau_1_seed_{}'.format(seed))
# tau 10
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_10_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_l2_tau_10_seed_{}'.format(seed))


### core 2
# tau 30
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_l2_tau_30_seed_{}'.format(seed))
# origin similarity, tau 1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --ddataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_1_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_origin_tau_1_seed_{}'.format(seed))


### core 3
# origin similarity, tau 10
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_10_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_origin_tau_10_seed_{}'.format(seed))
# origin similarity, tau 30
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_origin_tau_30_seed_{}'.format(seed))


### core 4
# shift 5_clusters, inverse_training_loss, tau 1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_invloss_tau_1_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_invloss_tau_1_seed_{}'.format(seed))
# shift 5_clusters, inverse_training_loss, tau 10
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_invloss_tau_10_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_invloss_tau_10_seed_{}'.format(seed))


### core 5
# shift 5_clusters, inverse_training_loss, tau 30
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_invloss_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_invloss_tau_30_seed_{}'.format(seed))
# shift 5_clusters, cosine_similarity, tau 1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_1_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_cosine_tau_1_seed_{}'.format(seed))

### core 6
# shift 5_clusters, cosine_similarity, tau 10
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_10_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_cosine_tau_10_seed_{}'.format(seed))
# shift 5_clusters, cosine_similarity, tau 30d
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_cosine_tau_30_seed_{}'.format(seed))


### core 7
# shift 5_clusters, l2, tau 1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_l2_tau_1_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_l2_tau_1_seed_{}'.format(seed))
# shift 5_clusters, l2, tau 10
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_l2_tau_10_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_l2_tau_10_seed_{}'.format(seed))


### core 8
# shift 5_clusters, l2, tau 30
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_l2_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_l2_tau_30_seed_{}'.format(seed))
# shift 5_clusters, cosine_origin, tau 1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_1_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_cosine_origin_tau_1_seed_{}'.format(seed))


### core 9
# shift 5_clusters, cosine_origin, tau 10
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_10_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_cosine_origin_tau_10_seed_{}'.format(seed))
# shift 5_clusters, cosine_origin, tau 30
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_DAC_cosine_origin_tau_30_seed_{}'.format(seed))


### core 10
# shift 5_clusters, no_exchange, invloss
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_invloss_seed_1 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(no_comm_lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_no_comm_invloss_seed_1')
# shift 5_clusters, no_exchange, cosine, seed 2
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_cosine_seed_2 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(no_comm_lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_no_comm_cosine_seed_2')
# shift 5_clusters, no_exchange, l2 , seed 3
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_l2_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(no_comm_lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_no_comm_l2_seed_3')
# shift 5_clusters, oracle, invloss, seed 1
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_oracle_invloss_seed_1 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_oracle_invloss_seed_1')
# shift 5_clusters, oracle, cosine, seed 2
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_oracle_cosine_seed_2 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_oracle_cosine_seed_2')
# shift 5_clusters, oracle, l2 , seed 3
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_oracle_l2_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_oracle_l2_seed_3')
# shift 5_clusters, random, invloss, seed 1
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_random_invloss_seed_1 --delusion -1.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_random_invloss_seed_1')
# shift 5_clusters, random, cosine, seed 2
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_random_cosine_seed_2 --delusion -1.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_random_cosine_seed_2')
# shift 5_clusters, random, l2 , seed 3
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_random_l2_seed_3 --delusion -1.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_5_clusters_random_l2_seed_3')


### core 11
# shift label, DAC, invloss, seed 3
taus = [1.0, 10.0, 30.0]
for tau in taus:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_{}_seed_3 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities False'.format(lr, tau, tau))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_invloss_tau_{}_seed_3'.format(tau))
# shift label, oracle, l2, seed 3
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_label_oracle_l2_seed_3 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_oracle_l2_seed_3')
# space for two



### core 12
# shift label, oracle, cosine, for all seeds, mergatron activate
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_label_oracle_cosine_seed_{}_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_oracle_cosine_seed_{}_MERGATRON'.format(seed))
# shift label, random, l2, for all seeds, mergatron activate
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --client_information_exchange oracle --experiment_name CIFAR_label_random_l2_seed_{}_MERGATRON --delusion -1.0 --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate'.format(seed, lr, seed))
    commands.append('python3 test_CIFAR.py --experiment CIFAR_label_random_l2_seed_{}_MERGATRON'.format(seed))



### core 11 extra runs
# shift label, DAC, invloss, seed 1, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_10_seed_1_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_invloss_tau_10_seed_1_MERGATRON')
# shift label, DAC, cosine, seed 1, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_10_seed_1_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_tau_10_seed_1_MERGATRON')
# shift label, DAC, l2, seed 1, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_10_seed_1_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_l2_tau_10_seed_1_MERGATRON')
# shift label, DAC, origin, seed 1, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_10_seed_1_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_origin_tau_10_seed_1_MERGATRON')


### core 12 extra runs
# shift label, DAC, cosine, seed 2, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_10_seed_2_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_tau_10_seed_2_MERGATRON')
# shift label, DAC, l2, seed 2, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_10_seed_2_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_l2_tau_10_seed_2_MERGATRON')
# shift label, DAC, origin, seed 2, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_10_seed_2_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_origin_tau_10_seed_2_MERGATRON')
# shift label, DAC, invloss, seed 2, mergatron, tau 10
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_10_seed_2_MERGATRON --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate'.format(lr))
commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_invloss_tau_10_seed_2_MERGATRON')


'''
'''

# misc 15/4 - tom
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.0001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_cosine_origin_seed_3 --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron chill --nbr_classes 10 --nbr_channels 3')
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_200_seed_3 --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron chill --nbr_classes 10 --nbr_channels 3')
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_200_seed_3 --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron chill --nbr_classes 10 --nbr_channels 3')

# seeds = [1, 2]
# for seed in seeds:
#     #commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
#     #commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_origin_tau_30_seed_{}'.format(seed))
#     pass

# misc 16/4 - tom
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_200_seed_3_MERGATRON --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron activate --nbr_classes 10 --nbr_channels 3')

### 16/4 - BIG RUN
'''
# inverseloss tau 5
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_5_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))

# # L2 tau 80, 100
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_80_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_200_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))




# # cosine tau 80, 100
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_80_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_200_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))

# # origin tau 30, 80, 100
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_origin_tau_30_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))

'''

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_origin_tau_80_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_origin_tau_200_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))


## 5 label shift 
# origin tau 30, 80
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_30_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))

'''

# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_80_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))

# # cosine tau 30, 80
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_30_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_80_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))


# # inversloss tau 5
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_invloss_tau_5_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))

# # l2 Tau 80, 200
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_l2_tau_80_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_l2_tau_200_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr))

'''

# origin 200
# cosine 200
# invloss 30, 10
# l2 30


### BIG RUN 16/4 LABELSHIFT - prior weighting for aggregation
# '''
# # cosine tau 30, 80, 200 core 6
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_30_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_80_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # cosine_origin tau 30, 80, 200 core 7
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_30_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_80_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # l2 tau 30, 80, 200 core 8
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_30_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_80_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_200_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # inverse tau 5, 10, 30 core 9
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_5_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_10_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_30_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))


### BIG RUN 16/4 TARGETBOX - prior weighting for aggregation

# # cosine tau 30, 80, 200
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_tau_30_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_tau_80_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_tau_200_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # cosine_origin tau 30, 80, 200
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_30_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_80_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_200_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # l2 tau 30, 80, 200
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_30_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_80_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_200_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # inverse tau 5, 10, 30
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_invloss_tau_5_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_invloss_tau_10_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_invloss_tau_30_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))



### FINAL TAU TUNING FOR CIFAR LABEL & 5 CLUSTERS ###

# 5 cluster cosine tau 300 and 500
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_300_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) #--aggregation_weighting priors

# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 500.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_500_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 


# # 5 cluster origin tau 300 and 500
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_300_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 

# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 500.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_500_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 


# # label cosine tau 300 and 500
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_300_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))

# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 500.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_500_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))


# # label origin tau 300 and 500
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_300_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))

# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 500.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_500_seed_2 --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(lr))


# MNIST DEV
# finding learning rate for MNIST with oracle 10^-4, 5*10^-5, 10^-5, 5*10^-6
#lr_tuning = 3e-4#1e-3 #7e-4 #3e-4 #1e-4 5e-5, 1e-5, 5e-6
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange oracle --experiment_name fashion_MNIST_oracle_tuning_lr_{}_seed_1 --delusion 0.0'.format(lr_tuning, lr_tuning))

# Label PRIORWEIGHTS
# invloss tau 1
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_1_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # l2 tau 10
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_10_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # cosine tau 80, 200, 300
# försenad :(
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_80_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# # ----
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_200_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_300_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# #origin tau 80, 200, 300
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 80.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_80_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# ----
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_200_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_300_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))


# 5 cluster PRIORWEIGHTS
# invloss tau 1
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_invloss_tau_1_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# # # ---

# # l2 tau 10
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 10.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_10_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # cosine tau 300, 500
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_tau_300_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# origin tau 300, 500
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 300.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_300_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))


# Core9 we need more runs for tau!
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0 --tau 500.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_tau_500_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0 --tau 500.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_500_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

timebomb_seconds = 60*60*8 # 8 hours
import time
#print('Sleeping for {} seconds = {} hours'.format(timebomb_seconds, timebomb_seconds/3600))
#time.sleep(timebomb_seconds)

# WE NEED EVEN MORE RUNS FOR TAU
# tau = 5, L2, priorweights, 5 cluster
# core15_tb
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_5_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_1_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# tau = 5, L2, priorweights, label
# core16_tb
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_5_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 1.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_1_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))

# # tau = 700, cosine, training weights, 5 cluster 
# core17_tb
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 700.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_700_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 

# # tau = 1000, cosine, training weights, 5 cluster 
# core18_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 1000.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_1000_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 


# MNIST TAUTUNING
# lr = 0.0003 # TBD
# no_comm_lr = 0.00005
# timebomb_seconds = 60*60*10 # 6 hours
# import time
# print('Sleeping for {} seconds'.format(timebomb_seconds))
# time.sleep(timebomb_seconds)

'''
core 10
lr = 0.0003 # TBD
'''
# # invloss trainingweight tau 5, 10, 30
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 5.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_invloss_tau_5_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_invloss_tau_10_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_invloss_tau_30_seed_1 --measure_all_similarities True'.format(lr))

# # core 11 
# # l2 tau 10, 30, 80
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_l2_tau_10_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_l2_tau_30_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_l2_tau_80_seed_1 --measure_all_similarities True'.format(lr))

# # core 12 
# # cosine tau 100, 200, 300
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 100.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_100_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_200_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_300_seed_1 --measure_all_similarities True'.format(lr))

# # core 13
# # origin tau 100, 200, 300
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 100.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_100_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_200_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_300_seed_1 --measure_all_similarities True'.format(lr))

# # invloss priorweight tau 5, 10, 30
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 5.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_invloss_tau_5_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_invloss_tau_10_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_invloss_tau_30_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))

# #core 5
# # l2 priorweight tau 10, 30, 80
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 10.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_l2_tau_10_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_l2_tau_30_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 80.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_l2_tau_80_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))


# #core 6
# # cosine priorweight tau 100, 200, 300
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 100.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_tau_100_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_tau_200_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_tau_300_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))

# '''
# # core 7
# # origin priorweight tau 100, 200, 300
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 100.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_origin_tau_100_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_origin_tau_200_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 300.0 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_origin_tau_300_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# '''

# # benchmarks random and no_comm and oracle
# # core 14
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 30.0 --client_information_exchange oracle --experiment_name fashion_MNIST_random_seed_1 --delusion -1.0'.format(no_comm_lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange no_exchange --experiment_name fashion_MNIST_no_comm_seed_1 --delusion 0.0'.format(no_comm_lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 30.0 --client_information_exchange oracle --experiment_name fashion_MNIST_oracle_seed_1 --delusion 0.0'.format(no_comm_lr))

# run seeds 1, 2, 3
seeds = [1, 2, 3]
lr = 0.001

'''
###### CIFAR LABELSHIFT TRAININGWEIGHT BEST TAU REPRODUCTION ########
label_invloss_tauopt = 10
label_l2_tauopt = 30
label_cosine_tauopt = 300
label_origin_tauopt = 200


# invloss - running on edvinbox core1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_invloss_tauopt, label_invloss_tauopt, seed))

# l2 - running on edvinbox core1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_l2_tauopt, label_l2_tauopt, seed))


# cosine - running on edvinbox core2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_cosine_tauopt, label_cosine_tauopt, seed))

# cosine origin - running on edvinbox core2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_origin_tauopt, label_origin_tauopt, seed))



###### CIFAR 5 CLUSTERS TRAININGWEIGHT BEST TAU REPRODUCTION ########
clusters_invloss_tauopt = 5
clusters_l2_tauopt = 30
clusters_cosine_tauopt = 1000
clusters_origin_tauopt = 300


# invloss - running on edvinbox core8
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_invloss_tauopt, clusters_invloss_tauopt, seed))

# l2 - running on edvinbox core8
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_l2_tauopt, clusters_l2_tauopt, seed))


# cosine - running on targetbox core1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_cosine_tauopt, clusters_cosine_tauopt, seed))

# cosine origin - running on targetbox core1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_origin_tauopt, clusters_origin_tauopt, seed))


###### CIFAR LABELSHIFT PRIORWEIGHT BEST TAU REPRODUCTION ########
label_priorweight_invloss_tauopt = 5
label_priorweight_l2_tauopt = 5
label_priorweight_cosine_tauopt = 200
label_priorweight_origin_tauopt = 200



# invloss - running on targetbox core2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_invloss_tauopt, label_priorweight_invloss_tauopt, seed))

# l2 - running on targetbox core2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_l2_tauopt, label_priorweight_l2_tauopt, seed))



# cosine - running on targetbox core3
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_cosine_tauopt, label_priorweight_cosine_tauopt, seed))

# cosine origin - running on targetbox core3
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_origin_tauopt, label_priorweight_origin_tauopt, seed))

'''

###### CIFAR 5 CLUSTERS PRIORWEIGHT BEST TAU REPRODUCTION ########
clusters_priorweight_invloss_tauopt = 5
clusters_priorweight_l2_tauopt = 10
clusters_priorweight_cosine_tauopt = 200
clusters_priorweight_origin_tauopt = 200

'''

# invloss - running on targetbox core4
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_invloss_tauopt, clusters_priorweight_invloss_tauopt, seed))

# l2 - running on targetbox core4
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_l2_tauopt, clusters_priorweight_l2_tauopt, seed))


# cosine - running on edvinbox core7
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_cosine_tauopt, clusters_priorweight_cosine_tauopt, seed))

# cosine origin - running on edvinbox core7
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_origin_tauopt, clusters_priorweight_origin_tauopt, seed))


###### CIFAR LABELSHIFT BEST TAU REPRODUCTION NUM EPOCHS 5 ########
label_invloss_tauopt = 10
label_l2_tauopt = 30
label_cosine_tauopt = 300
label_origin_tauopt = 200

seeds = [1, 2, 3]
lr = 0.001

# invloss - running on targetbox core2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_invloss_tauopt, label_invloss_tauopt, seed))

# l2 - running on targetbox core4
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_l2_tauopt, label_l2_tauopt, seed))

# cosine - running on edvinbox core7
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_cosine_tauopt, label_cosine_tauopt, seed))

# cosine origin - running on edvinbox core8
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_origin_tauopt, label_origin_tauopt, seed))
'''
###### CIFAR LABELSHIFT PRIORWEIGHT BEST TAU REPRODUCTION NUM EPOCHS 5 ########
label_priorweight_invloss_tauopt = 5
label_priorweight_l2_tauopt = 5
label_priorweight_cosine_tauopt = 200
label_priorweight_origin_tauopt = 200

seeds = [1, 2, 3]
lr = 0.001
'''
# invloss - running on edvinbox core2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_invloss_tauopt, label_priorweight_invloss_tauopt, seed))

# l2 - running on edvinbox core3
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_l2_tauopt, label_priorweight_l2_tauopt, seed))

# cosine - running on edvinbox core4
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_cosine_tauopt, label_priorweight_cosine_tauopt, seed))

# cosine origin - running on edvinbox core5
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_origin_tauopt, label_priorweight_origin_tauopt, seed))


###### CIFAR LABELSHIFT BEST TAU REPRODUCTION NUM EPOCHS 10 ########
label_invloss_tauopt = 10
label_l2_tauopt = 30
label_cosine_tauopt = 300
label_origin_tauopt = 200

seeds = [1, 2, 3]
lr = 0.001

# invloss - running on edvinbox core1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_invloss_tauopt, label_invloss_tauopt, seed))

# l2 - running on edvinbox core3
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_l2_tauopt, label_l2_tauopt, seed))

# cosine seed 1 and 2 - running on edvinbox core9
for seed in [1,2]:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_cosine_tauopt, label_cosine_tauopt, seed))

#cosine seed 3 - running on edvinbox core2
seed = 3
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_cosine_tauopt, label_cosine_tauopt, seed))

# cosine origin seed 1 - running on edvinbox core2
for seed in [1]:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_origin_tauopt, label_origin_tauopt, seed))

# cosine origin seeds 2 and 3 - running on edvinbox core4
for seed in [2,3]:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_origin_tauopt, label_origin_tauopt, seed))
'''
###### CIFAR LABELSHIFT PRIORWEIGHT BEST TAU REPRODUCTION NUM EPOCHS 10 ########
label_priorweight_invloss_tauopt = 5
label_priorweight_l2_tauopt = 5
label_priorweight_cosine_tauopt = 200
label_priorweight_origin_tauopt = 200

seeds = [1, 2, 3]
lr = 0.001
'''
# invloss - running on edvinbox core6
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_invloss_tauopt, label_priorweight_invloss_tauopt, seed))

# l2 - running on edvinbox core7
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_l2_tauopt, label_priorweight_l2_tauopt, seed))

# cosine - running on edvinbox core8
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_cosine_tauopt, label_priorweight_cosine_tauopt, seed))

# cosine origin - running on edvinbox core9
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_origin_tauopt, label_priorweight_origin_tauopt, seed))


    
###### BENCHMARK FASHON MNIST ######## - running on targetbox core1
no_comm_lr = 0.00005 # check this with existing run to determine viability
lr = 0.0003

seeds = [2, 3]
# no comm
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange no_exchange --experiment_name fashion_MNIST_no_comm_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, no_comm_lr, seed))

# random
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange oracle --experiment_name fashion_MNIST_random_seed_{} --delusion -1.0 --measure_all_similarities True'.format(seed, no_comm_lr, seed))

# oracle
seed = 3
commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange oracle --experiment_name fashion_MNIST_oracle_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, no_comm_lr, seed))

# oracle seed 2
seed = 2
commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange oracle --experiment_name fashion_MNIST_oracle_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, no_comm_lr, seed))

'''

###### FASHION MNIST TRAININGWEIGHT BEST TAU REPRODUCTION ########
mnist_invloss_tauopt = 10
mnist_l2_tauopt = 10
mnist_cosine_tauopt = 2000
mnist_origin_tauopt = 2000

seeds = [2, 3]
# lrs from above
no_comm_lr = 0.00005 # check this with existing run to determine viability
lr = 0.0003
'''
# invloss - running on edvinbox core6
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_invloss_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_invloss_tauopt, mnist_invloss_tauopt, seed))

# l2 - running on edvinbox core6
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_l2_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_l2_tauopt, mnist_l2_tauopt, seed))


# cosine - running on edvinbox core3
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_cosine_tauopt, mnist_cosine_tauopt, seed))

# cosine origin - running on edvinbox core4
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_origin_tauopt, mnist_origin_tauopt, seed))

###### FASHION MNIST PRIORWEIGHT BEST TAU REPRODUCTION ########
mnist_priorweight_invloss_tauopt = 5 
mnist_priorweight_l2_tauopt = 10
mnist_priorweight_cosine_tauopt = 300
mnist_priorweight_origin_tauopt = 300

seeds = [2, 3]
# lrs from above
no_comm_lr = 0.00005 # check this with existing run to determine viability
lr = 0.0003

# invloss - running on edvinbox core5
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_invloss_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_invloss_tauopt, mnist_priorweight_invloss_tauopt, seed))

# l2 - running on edvinbox core5
for seed in seeds: 
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_l2_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_l2_tauopt, mnist_priorweight_l2_tauopt, seed))

# cosine - running on edvinbox core5
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_cosine_tauopt, mnist_priorweight_cosine_tauopt, seed))


# cosine origin - running on edvinbox core6
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_origin_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_origin_tauopt, mnist_priorweight_origin_tauopt, seed))


##### MISC EXTRA RUNS #####

# for ???
lr = 0.001
no_comm_lr = 0.0001

### CIFAR ###
# CORE1 -------
# cifar 5 clusters trainingweight cosine tau 2000
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 2000.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_2000_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 

# CORE2 -------
# cifar 5 clusters trainingweight cosine tau 5000
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 5000.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_5000_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 

# # cifar 5 clusters trainingweight cosine tau 10000
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 10000.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_10000_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 

lr = 0.0003
### MNIST ###
# CORE3 ------
# MNIST L2 priorweights tau 5, tau 1
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau 5 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_l2_tau_5_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau 1 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_l2_tau_1_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))

# MNIST invloss priorweights tau 1
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 1 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_invloss_tau_1_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))

# CORE4 ------
# MNIST Origin priorweights tau 500, tau 2000
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau 500 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_origin_tau_500_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau 2000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_origin_tau_2000_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))

# MNIST cosine priorweights tau 500, tau 2000
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 500 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_tau_500_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))

# CORE5 ------
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 2000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_tau_2000_seed_1 --aggregation_weighting priors --measure_all_similarities True'.format(lr))

# MNIST L2 trainingweights tau 5, tau 1
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau 5 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_l2_tau_5_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau 1 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_l2_tau_1_seed_1 --measure_all_similarities True'.format(lr))




# MNIST Origin trainingweights tau 50, tau 500, tau 2000
# CORE6 ------
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau 50 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_50_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau 500 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_500_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau 2000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_2000_seed_1 --measure_all_similarities True'.format(lr))

# MNIST Cosine trainingweights tau 500, tau 2000
# CORE7 ------
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 500 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_500_seed_1 --measure_all_similarities True'.format(lr))
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 2000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_2000_seed_1 --measure_all_similarities True'.format(lr))


'''
# for mnist
#no_comm_lr = 0.00005 # check this with existing run to determine viability
#lr = 0.0003

### More MNIST tautuning ###
# cosine origin tau 5000 - running on edvinbox core3
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau 5000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_5000_seed_1 --measure_all_similarities True'.format(lr))
# cosine origin tau 10000 - running on edvinbox core3
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau 10000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_10000_seed_1 --measure_all_similarities True'.format(lr))


# cosine tau 5000 - running on edvinbox core4
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 5000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_5000_seed_1 --measure_all_similarities True'.format(lr))
# cosine tau 10000 - running on edvinbox core4
#commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 10000 --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_10000_seed_1 --measure_all_similarities True'.format(lr))


# ##### CIFAR BENCHMARKS RERUNS #####
# lr_no_comm = 0.0001
# lr = 0.001

# cifar label no comm seed 1 - running on edvinbox core4_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange no_exchange --experiment_name CIFAR_label_no_comm_seed_1_fixed --delusion 0.0 --measure_all_similarities True'.format(lr_no_comm))
# cifar label no comm seed 2 - running on edvinbox core4_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange no_exchange --experiment_name CIFAR_label_no_comm_seed_2_fixed --delusion 0.0 --measure_all_similarities True'.format(lr_no_comm))
# cifar label no comm seed 3 - running on edvinbox core4_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange no_exchange --experiment_name CIFAR_label_no_comm_seed_3_fixed --delusion 0.0 --measure_all_similarities True'.format(lr_no_comm))

# cifar label oracle seed 1 - running on edvinbox core1_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_label_oracle_seed_1_fixed --delusion 0.0 --measure_all_similarities False'.format(lr))
# cifar label oracle seed 2 - running on edvinbox core1_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_label_oracle_seed_2_fixed --delusion 0.0 --measure_all_similarities False'.format(lr))
# cifar label oracle seed 3 - running on edvinbox core1_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_label_oracle_seed_3_fixed --delusion 0.0 --measure_all_similarities False'.format(lr))

# cifar label random seed 1 - running on edvinbox core3_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_label_random_seed_1_fixed --delusion -1.0 --measure_all_similarities False'.format(lr))
# cifar label random seed 2 - running on edvinbox core3_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_label_random_seed_2_fixed --delusion -1.0 --measure_all_similarities False'.format(lr))
# cifar label random seed 3 - running on edvinbox core3_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_label_random_seed_3_fixed --delusion -1.0 --measure_all_similarities False'.format(lr))

# cifar 5 clusters no comm seed 1 - running on edvinbox core4_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_seed_1_fixed --delusion 0.0 --measure_all_similarities True'.format(lr_no_comm))
# cifar 5 clusters no comm seed 2 - running on edvinbox core4_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_seed_2_fixed --delusion 0.0 --measure_all_similarities True'.format(lr_no_comm))
# cifar 5 clusters no comm seed 3 - running on edvinbox core4_tb
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_seed_3_fixed --delusion 0.0 --measure_all_similarities True'.format(lr_no_comm))

# cifar 5 clusters oracle seed 1 - running on edvinbox core7
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_oracle_seed_1_fixed --delusion 0.0 --measure_all_similarities False'.format(lr))
# cifar 5 clusters oracle seed 2 - running on edvinbox core7
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_oracle_seed_2_fixed --delusion 0.0 --measure_all_similarities False'.format(lr))
# cifar 5 clusters oracle seed 3 - running on edvinbox core7
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_oracle_seed_3_fixed --delusion 0.0 --measure_all_similarities False'.format(lr))

# cifar 5 clusters random seed 1 - running on edvinbox core8
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_random_seed_1_fixed --delusion -1.0 --measure_all_similarities False'.format(lr))
# cifar 5 clusters random seed 2 - running on edvinbox core8
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_random_seed_2_fixed --delusion -1.0 --measure_all_similarities False'.format(lr))
# cifar 5 clusters random seed 3 - running on edvinbox core8
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange oracle --experiment_name CIFAR_5_clusters_random_seed_3_fixed --delusion -1.0 --measure_all_similarities False'.format(lr))

# for mnist
#lr = 0.0003

## TARGET_BOX timebomb core5_tb 9 hour, 17:49
seeds = [2,3]
# Oracle fashion_MNIST
#for seed in seeds:
#    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange oracle --experiment_name fashion_MNIST_oracle_seed_{}_fixed --delusion 0.0 --measure_all_similarities False'.format(seed, lr, seed))

## TARGET_BOX timebomb core6_tb 8 hour, 17:49
# seeds = [1,2,3]
# # Random fashion_MNIST
# for seed in seeds:
#     commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange oracle --experiment_name fashion_MNIST_random_seed_{}_fixed --delusion -1.0 --measure_all_similarities False'.format(seed, lr, seed))


##### TOYPROBLEM TAUTUNING #####
import numpy as np
seed = 1
lrs = np.logspace(np.log10(0.008), np.log10(0.0003), num=4)
taus_trainingweight_invloss1 = np.logspace(np.log10(50000), np.log10(5000000), num=8)
taus_trainingweight_invloss2 = np.logspace(np.log10(10), np.log10(10000), num=8)
# concat the arrays
taus_trainingweight_invloss = np.logspace(np.log10(1), np.log(9), num=8)#np.concatenate((taus_trainingweight_invloss1, taus_trainingweight_invloss2))
taus_trainingweight_l2 = np.logspace(np.log10(2), np.log10(100), num=8)
taus_trainingweight_cosine = np.logspace(np.log10(10), np.log10(1000), num=8)
taus_trainingweight_cosine_origin = np.logspace(np.log10(10), np.log10(1000), num=8)
taus_priorweight_invloss1 = np.logspace(np.log10(50000), np.log10(5000000), num=8)
taus_priorweight_invloss2 = np.logspace(np.log10(10), np.log10(10000), num=8)
# concat the arrays
taus_priorweight_invloss = np.concatenate((taus_priorweight_invloss1, taus_priorweight_invloss2))
taus_priorweight_l2 = np.logspace(np.log10(2), np.log10(500), num=8)
taus_priorweight_cosine = np.logspace(np.log10(50), np.log10(5000), num=8)
taus_priorweight_cosine_origin = np.logspace(np.log10(50), np.log10(5000), num=8)
# first two lrs on targetbox core1, next two on core2
'''
for lr in lrs:
    
    # invloss
    for tau in taus_trainingweight_invloss:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_invloss_lr_{}_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr, tau, lr, tau, seed))
    
    # l2
    for tau in taus_trainingweight_l2:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_l2_lr_{}_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr, tau, lr, tau, seed))

    # cosine
    for tau in taus_trainingweight_cosine:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_cosine_lr_{}_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr, tau, lr, tau, seed))

    # cosine origin
    for tau in taus_trainingweight_cosine_origin:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_cosine_origin_lr_{}_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr, tau, lr, tau, seed))
    
    # invloss
    for tau in taus_priorweight_invloss:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_invloss_prior_lr_{}_tau_{}_seed_{} --delusion 0.0 --aggregation_weighting priors'.format(seed, lr, tau, lr, tau, seed))
    
    # l2
    for tau in taus_priorweight_l2:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_l2_prior_lr_{}_tau_{}_seed_{} --delusion 0.0 --aggregation_weighting priors'.format(seed, lr, tau, lr, tau, seed))

    # cosine
    for tau in taus_priorweight_cosine:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_cosine_lr_{}_tau_{}_seed_{} --delusion 0.0 --aggregation_weighting priors'.format(seed, lr, tau, lr, tau, seed))

    # cosine origin
    for tau in taus_priorweight_cosine_origin:
        commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_cosine_origin_lr_{}_tau_{}_seed_{} --delusion 0.0 --aggregation_weighting priors'.format(seed, lr, tau, lr, tau, seed))

    # benchmarks
    # oracle
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange oracle --experiment_name TOY_oracle_lr_{}_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, lr, lr, seed))
    # random
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange oracle --experiment_name TOY_random_lr_{}_seed_{} --delusion -1.0 --measure_all_similarities True'.format(seed, lr, lr, seed))
    # no comm
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange no_exchange --experiment_name TOY_no_comm_lr_{}_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, lr, lr, seed))


##### TOYPROBLEM BENCHMARKS ##### - running on targetbox core3_tb

lr = 7.5e-05
no_comm_lr = 7.5e-06
seeds = [1,2,3]

# oracle
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange oracle --experiment_name TOY_oracle_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, lr, seed))

# random
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange oracle --experiment_name TOY_random_seed_{} --delusion -1.0 --measure_all_similarities True'.format(seed, lr, seed))

# no comm
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange no_exchange --experiment_name TOY_no_comm_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, no_comm_lr, seed))



##### TOYPROBLEM REPRODUCTION #####
toy_trainingweight_invloss_tauopt = 10000
toy_trainingweight_l2_tauopt = 19
toy_trainingweight_cosine_tauopt = 140
toy_trainingweight_cosine_origin_tauopt = 140
toy_priorweight_invloss_tauopt = 5000
toy_priorweight_l2_tauopt = 19
toy_priorweight_cosine_tauopt = 140
toy_priorweight_cosine_origin_tauopt = 140

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
lr = 0.003
lr_invloss_l2 = 0.008

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_invloss_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, lr_invloss_l2, toy_trainingweight_invloss_tauopt, toy_trainingweight_invloss_tauopt, seed))

# l2 - running on targetbox core1
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_l2_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, lr_invloss_l2, toy_trainingweight_l2_tauopt, toy_trainingweight_l2_tauopt, seed))

# cosine - running on targetbox core2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_cosine_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, lr, toy_trainingweight_cosine_tauopt, toy_trainingweight_cosine_tauopt, seed))

# cosine origin - running on targetbox core3
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_cosine_origin_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True'.format(seed, lr, toy_trainingweight_cosine_origin_tauopt, toy_trainingweight_cosine_origin_tauopt, seed))

# invloss priorweight
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_invloss_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr_invloss_l2, toy_priorweight_invloss_tauopt, toy_priorweight_invloss_tauopt, seed))

# l2 priorweight - running on targetbox core4
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_l2_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, toy_priorweight_l2_tauopt, toy_priorweight_l2_tauopt, seed))

# cosine priorweight
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_cosine_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, toy_priorweight_cosine_tauopt, toy_priorweight_cosine_tauopt, seed))

# cosine origin priorweight
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_priorweight_cosine_origin_tau_{}_seed_{} --delusion 0.0 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, toy_priorweight_cosine_origin_tauopt, toy_priorweight_cosine_origin_tauopt, seed))
'''



##### TOYPROBLEM MINMAX REPRODUCTION #####
toy_trainingweight_invloss_tauopt = 10
toy_trainingweight_l2_tauopt = 10
toy_trainingweight_cosine_tauopt = 100
toy_trainingweight_cosine_origin_tauopt = 100
toy_priorweight_invloss_tauopt = 5000
toy_priorweight_l2_tauopt = 19
toy_priorweight_cosine_tauopt = 140
toy_priorweight_cosine_origin_tauopt = 140

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lr = 0.003
lr_invloss_l2 = 0.008
'''
# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_invloss_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr_invloss_l2, toy_trainingweight_invloss_tauopt, toy_trainingweight_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_l2_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr, toy_trainingweight_l2_tauopt, toy_trainingweight_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_cosine_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr, toy_trainingweight_cosine_tauopt, toy_trainingweight_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset toy_problem --nbr_rounds 40 --nbr_clients 99 --n_data_train 50 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau {} --client_information_exchange DAC --experiment_name TOY_DAC_minmax_cosine_origin_tau_{}_seed_{} --delusion 0.0 --minmax True'.format(seed, lr, toy_trainingweight_cosine_origin_tauopt, toy_trainingweight_cosine_origin_tauopt, seed))



# testing
commands.append('python3 test_multiple.py')

# core6
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange no_exchange --experiment_name fashion_MNIST_no_comm_seed_2 --delusion 0.0 --NAEM_frequency 5 --T1 50 --nbr_deluded_clients 0 --mergatron chill --aggregation_weighting trainset_size --nbr_classes 10 --nbr_channels 3')

# Core9
# commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0.0 --tau 30.0 --client_information_exchange no_exchange --experiment_name fashion_MNIST_no_comm_seed_3 --delusion 0.0 --NAEM_frequency 5 --T1 50 --nbr_deluded_clients 0 --mergatron chill --aggregation_weighting trainset_size --nbr_classes 10 --nbr_channels 3')
'''


##### CIFAR100 TAUTUNING (Pretrained) #####
invloss_taus = np.logspace(np.log10(1), np.log10(150), num=6)
l2_taus = np.logspace(np.log10(1), np.log10(150), num=6)
cosine_taus = np.logspace(np.log10(10), np.log10(300), num=6)
cosine_origin_taus = np.logspace(np.log10(10), np.log10(300), num=6)
'''
# invloss - running on edvinbox core1
for tau in invloss_taus:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar100 --shift label --nbr_rounds 270 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name HUNDRED_pretrained_invloss_tau_{} --delusion 0.0 --measure_all_similarities True --model pretrained'.format(tau, tau))

# l2 - running on edvinbox core2
for tau in l2_taus:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar100 --shift label --nbr_rounds 270 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name HUNDRED_pretrained_l2_tau_{} --delusion 0.0 --measure_all_similarities True --model pretrained'.format(tau, tau))

# cosine - running on edvinbox core3
for tau in cosine_taus:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar100 --shift label --nbr_rounds 270 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name HUNDRED_pretrained_cosine_tau_{} --delusion 0.0 --measure_all_similarities True --model pretrained'.format(tau, tau))

# cosine origin - running on edvinbox core4
for tau in cosine_origin_taus:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar100 --shift label --nbr_rounds 270 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name HUNDRED_pretrained_cosine_origin_tau_{} --delusion 0.0 --measure_all_similarities True --model pretrained'.format(tau, tau))
'''
# commands.append('python3 test_multiple.py')


##### CIFAR100 lr-tuning (nonpretrained) #####
lrs = np.logspace(np.log10(7.5e-05), np.log10(1e-03), num=5)

# oracle tautune
for lr in lrs[2:]:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar100 --shift label --nbr_rounds 270 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange oracle --experiment_name HUNDRED_nonpretrained_oracle_lr_{}_seed_1 --delusion 0.0 --measure_all_similarities True --model nonpretrained'.format(lr, lr))

# no comm
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar100 --shift label --nbr_rounds 270 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange no_exchange --experiment_name HUNDRED_nonpretrained_no_comm_lr_7.5e-05_seed_1 --delusion 0.0 --measure_all_similarities True --model nonpretrained')
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar100 --shift label --nbr_rounds 270 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30 --client_information_exchange no_exchange --experiment_name HUNDRED_pretrained_no_comm_lr_7.5e-05_seed_1 --delusion 0.0 --measure_all_similarities True --model pretrained')


print('Commands to be run: ')
for command in commands:
    print(command)

print('')
print('Running {} commands'.format(len(commands)))

# timebomb sleep for 1 hour
sleeptime = 0
sleeptime = 60*60*0.01
print('Sleeping for {} seconds'.format(sleeptime))
time.sleep(sleeptime)

for command in commands:
    subprocess.run(command, shell=True)

    