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

lr = 0.001
no_comm_lr = 0.0001

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

### BIG SETUP 12/4/2024 ###

# CORE 1
'''
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_1_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_1_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_1_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_1_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_1_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_1_seed_3_MERGATRON')


commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_10_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_10_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_10_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_10_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_10_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_10_seed_3_MERGATRON')


# CORE 2

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_30_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_30_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_30_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_30_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_invloss_tau_30_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_invloss_tau_30_seed_3_MERGATRON')

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_1_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_1_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_1_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_1_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_1_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_1_seed_3_MERGATRON')

# CORE 3
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_10_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_10_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_10_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_10_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_10_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_10_seed_3_MERGATRON')

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_30_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_30_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_30_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_30_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_cosine_tau_30_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_30_seed_3_MERGATRON')

# CORE 4
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_1_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_1_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_1_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_1_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_1_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_1_seed_3_MERGATRON')

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_10_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_10_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_10_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_10_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_10_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_10_seed_3_MERGATRON')

# CORE 5

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_30_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_30_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_30_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_30_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric l2 --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_l2_tau_30_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_l2_tau_30_seed_3_MERGATRON')

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_1_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_1_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_1_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_1_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_1_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_1_seed_3_MERGATRON')

# CORE 6
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_10_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_10_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_10_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_10_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_10_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_10_seed_3_MERGATRON')

commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_30_seed_1_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_30_seed_1_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_30_seed_2_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_30_seed_2_MERGATRON')
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_origin --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --mergatron activate --experiment_name CIFAR_LABEL_DAC_origin_tau_30_seed_3_MERGATRON')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_origin_tau_30_seed_3_MERGATRON')
'''

### 15 april 2024 6 cosine runs
# CORE 1
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --experiment_name CIFAR_LABEL_DAC_cosine_tau_1_seed_1')

# CORE 2
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --experiment_name CIFAR_LABEL_DAC_cosine_tau_10_seed_1')

# # CORE 3
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --experiment_name CIFAR_LABEL_DAC_cosine_tau_30_seed_1')

# # CORE 4
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 1.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --experiment_name CIFAR_LABEL_DAC_cosine_tau_1_seed_3')

# # CORE 5
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 10.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities True --experiment_name CIFAR_LABEL_DAC_cosine_tau_10_seed_3')

# CORE 6 RUN AGAIN!
# commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange no_exchange --experiment_name CIFAR_invloss_no_comm_seed_1 --CIFAR_ratio 0.4 --measure_all_similarities True'.format(no_comm_lr))
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift label --nbr_rounds 300 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 3 --batch_size 8 --nbr_local_epochs 1 --lr 0.001 --stopping_rounds 50 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric cosine_similarity --tau 30.0 --client_information_exchange DAC --CIFAR_ratio 0.4 --measure_all_similarities False --experiment_name CIFAR_LABEL_DAC_cosine_tau_30_seed_3')
commands.append('python3 test_CIFAR.py --experiment CIFAR_invloss_no_comm_seed_3')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_30_seed_1')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_1_seed_1')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_10_seed_1')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_30_seed_1')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_1_seed_3')
commands.append('python3 test_CIFAR.py --experiment CIFAR_LABEL_DAC_cosine_tau_10_seed_1')


for command in commands:
    subprocess.run(command, shell=True)