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

# WE NEED EVEN MORE RUNS FOR TAU
# tau = 5, L2, priorweights, 5 cluster
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_5_seed_3 --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
## tau = 5, L2, priorweights, label
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --cosine_alpha 0 --tau 5.0 --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_5_seed_2 --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(lr))
# tau = 800, cosine, training weights, 5 cluster 
#commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --cosine_alpha 0.0 --tau 800.0 --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_800_seed_3 --delusion 0.0 --CIFAR_ratio 0.2 --measure_all_similarities True'.format(lr)) 



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

'''

###### CIFAR LABELSHIFT TRAININGWEIGHT BEST TAU REPRODUCTION ########
label_invloss_tauopt = 10
label_l2_tauopt = 30
label_cosine_tauopt = 300
label_origin_tauopt = 200
 
# run seeds 1, 2, 3
seeds = [1, 2, 3]
lr = 0.001

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_invloss_tauopt, label_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_l2_tauopt, label_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_cosine_tauopt, label_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_origin_tauopt, label_origin_tauopt, seed))


###### CIFAR 5 CLUSTERS TRAININGWEIGHT BEST TAU REPRODUCTION ########
clusters_invloss_tauopt = 5
clusters_l2_tauopt = 30
clusters_cosine_tauopt = 0 # not final determine before run
clusters_origin_tauopt = 300 # maybe final check again

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_invloss_tauopt, clusters_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_l2_tauopt, clusters_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_cosine_tauopt, clusters_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True'.format(seed, lr, clusters_origin_tauopt, clusters_origin_tauopt, seed))


###### CIFAR LABELSHIFT PRIORWEIGHT BEST TAU REPRODUCTION ########
label_priorweight_invloss_tauopt = 5
label_priorweight_l2_tauopt = 0 # not final determine before run
label_priorweight_cosine_tauopt = 0 # not final determine before run
label_priorweight_origin_tauopt = 0 # not final determine before run

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_invloss_tauopt, label_priorweight_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_l2_tauopt, label_priorweight_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_cosine_tauopt, label_priorweight_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_origin_tauopt, label_priorweight_origin_tauopt, seed))


###### CIFAR 5 CLUSTERS PRIORWEIGHT BEST TAU REPRODUCTION ########
clusters_priorweight_invloss_tauopt = 0 # not final determine before run
clusters_priorweight_l2_tauopt = 0 # not final determine before run
clusters_priorweight_cosine_tauopt = 200
clusters_priorweight_origin_tauopt = 0 # not final determine before run

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_invloss_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_invloss_tauopt, clusters_priorweight_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_l2_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_l2_tauopt, clusters_priorweight_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_cosine_tauopt, clusters_priorweight_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift 5_clusters --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_5_clusters_DAC_priorweight_cosine_origin_tau_{}_seed_{}_fixed --CIFAR_ratio 0.2 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, clusters_priorweight_origin_tauopt, clusters_priorweight_origin_tauopt, seed))

###### CIFAR LABELSHIFT BEST TAU REPRODUCTION NUM EPOCHS 5 ########
# take optimal taus from above

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_invloss_tauopt, label_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_l2_tauopt, label_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_cosine_tauopt, label_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_origin_tauopt, label_origin_tauopt, seed))


###### CIFAR LABELSHIFT PRIORWEIGHT BEST TAU REPRODUCTION NUM EPOCHS 5 ########
# take optimal taus from above

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_invloss_tauopt, label_priorweight_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_l2_tauopt, label_priorweight_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_cosine_tauopt, label_priorweight_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 5 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_{}_seed_{}_5epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_origin_tauopt, label_priorweight_origin_tauopt, seed))


###### CIFAR LABELSHIFT BEST TAU REPRODUCTION NUM EPOCHS 10 ########
# take optimal taus from above

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_invloss_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_invloss_tauopt, label_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_l2_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_l2_tauopt, label_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_cosine_tauopt, label_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_cosine_origin_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True'.format(seed, lr, label_origin_tauopt, label_origin_tauopt, seed))


###### CIFAR LABELSHIFT PRIORWEIGHT BEST TAU REPRODUCTION NUM EPOCHS 10 ########
# take optimal taus from above

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_invloss_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_invloss_tauopt, label_priorweight_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_l2_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_l2_tauopt, label_priorweight_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_cosine_tauopt, label_priorweight_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 10 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name CIFAR_label_DAC_priorweight_cosine_origin_tau_{}_seed_{}_10epochs --CIFAR_ratio 0.4 --measure_all_similarities True --aggregation_weighting priors'.format(seed, lr, label_priorweight_origin_tauopt, label_priorweight_origin_tauopt, seed))


###### BENCHMARK FASHON MNIST ########
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


###### FASHION MNIST TRAININGWEIGHT BEST TAU REPRODUCTION ########
mnist_invloss_tauopt = 0 # not final determine before run
mnist_l2_tauopt = 0 # not final determine before run
mnist_cosine_tauopt = 0 # not final determine before run
mnist_origin_tauopt = 0 # not final determine before run

seeds = [2, 3]
# lrs from above

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_invloss_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_invloss_tauopt, mnist_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_l2_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_l2_tauopt, mnist_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_cosine_tauopt, mnist_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_cosine_origin_tau_{}_seed_{} --measure_all_similarities True'.format(seed, lr, mnist_origin_tauopt, mnist_origin_tauopt, seed))


###### FASHION MNIST PRIORWEIGHT BEST TAU REPRODUCTION ########
mnist_priorweight_invloss_tauopt = 0 # not final determine before run
mnist_priorweight_l2_tauopt = 0 # not final determine before run
mnist_priorweight_cosine_tauopt = 0 # not final determine before run
mnist_priorweight_origin_tauopt = 0 # not final determine before run

seeds = [2, 3]
# lrs from above

# invloss
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_invloss_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_invloss_tauopt, mnist_priorweight_invloss_tauopt, seed))

# l2
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric l2 --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_l2_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_l2_tauopt, mnist_priorweight_l2_tauopt, seed))

# cosine
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_similarity --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_cosine_tauopt, mnist_priorweight_cosine_tauopt, seed))

# cosine origin
for seed in seeds:
    commands.append('python3 run_experiment.py --gpu 0 --dataset fashion_mnist --nbr_rounds 300 --nbr_clients 100 --n_data_train 500 --n_data_val 100 --seed {} --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 4 --prior_update_rule softmax --similarity_metric cosine_origin --tau {} --client_information_exchange DAC --experiment_name fashion_MNIST_DAC_priorweight_cosine_origin_tau_{}_seed_{} --aggregation_weighting priors --measure_all_similarities True'.format(seed, lr, mnist_priorweight_origin_tauopt, mnist_priorweight_origin_tauopt, seed))

'''


for command in commands:
    subprocess.run(command, shell=True)