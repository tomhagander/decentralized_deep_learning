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

# misc 15/4 - tom
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.0001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_5_clusters_no_comm_cosine_origin_seed_3 --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron chill --nbr_classes 10 --nbr_channels 3')
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_200_seed_3 --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron chill --nbr_classes 10 --nbr_channels 3')
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_200_seed_3 --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron chill --nbr_classes 10 --nbr_channels 3')

seeds = [1, 2]
for seed in seeds:
    #commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --cosine_alpha 0.0 --tau 30.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_30_seed_{} --delusion 0.0 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, seed))
    #commands.append('python3 test_CIFAR.py --experiment CIFAR_label_DAC_cosine_origin_tau_30_seed_{}'.format(seed))
    pass

# misc 16/4 - tom
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 200.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_200_seed_3_MERGATRON --delusion 0.0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.2 --nbr_deluded_clients 0 --measure_all_similarities True --mergatron activate --nbr_classes 10 --nbr_channels 3')


for command in commands:
    subprocess.run(command, shell=True)