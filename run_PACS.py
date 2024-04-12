import subprocess

commands = []

# local benchmarks
# no communication
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --client_information_exchange no_exchange --experiment_name PACS_BENCH_NO_COMM')
# 4 clients, same thing
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 4 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --client_information_exchange no_exchange --experiment_name PACS_BENCH_ALL_DATA')
# random comms
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --client_information_exchange oracle --delusion -1 --experiment_name PACS_BENCH_RANDOM_COMM')

# some delusion
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --client_information_exchange some_delusion --nbr_deluded_clients 1 --experiment_name PACS_BENCH_1_DISSIDENCE')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --client_information_exchange some_delusion --nbr_deluded_clients 5 --experiment_name PACS_BENCH_5_DISSIDENCE')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --client_information_exchange some_delusion --nbr_deluded_clients 13 --experiment_name PACS_BENCH_13_DISSIDENCE')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --client_information_exchange some_delusion --nbr_deluded_clients 25 --experiment_name PACS_BENCH_25_DISSIDENCE')

# algos
# test DAC
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --prior_update_rule softmax-variable-tau --tau 30 --similarity_metric inverse_training_loss --client_information_exchange DAC --experiment_name PACS_DAC_VAR_INV')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --prior_update_rule softmax-variable-tau --tau 30 --similarity_metric cosine_similarity --cosine_alpha 0 --client_information_exchange DAC --experiment_name PACS_DAC_VAR_COS_0')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --prior_update_rule softmax-variable-tau --tau 30 --similarity_metric cosine_similarity --cosine_alpha 0.5 --client_information_exchange DAC --experiment_name PACS_DAC_VAR_COS_0.5')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --prior_update_rule softmax-variable-tau --tau 30 --similarity_metric cosine_similarity --cosine_alpha 1 --client_information_exchange DAC --experiment_name PACS_DAC_VAR_COS_1')

# test PANMd
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --similarity_metric inverse_training_loss --client_information_exchange PANM --NAEM_frequency 2 --T1 20 --experiment_name PACS_PANM_INV')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --similarity_metric cosine_similarity --cosine_alpha 0 --client_information_exchange PANM --NAEM_frequency 2 --T1 20 --experiment_name PACS_PANM_COS_0')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --similarity_metric cosine_similarity --cosine_alpha 0.5 --client_information_exchange PANM --NAEM_frequency 2 --T1 20 --experiment_name PACS_PANM_COS_0.5')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --seed 42 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 5 --similarity_metric cosine_similarity --cosine_alpha 1 --client_information_exchange PANM --NAEM_frequency 2 --T1 20 --experiment_name PACS_PANM_COS_1')

#### 40 Clients ####
# DAC with cosine similarity and inverse training loss, cosine origin and l2
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 3 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name PACS_40_clients_DAC_cosine_initial_3')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 3 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric inverse_training_loss --measure_all_similarities True --experiment_name PACS_40_clients_DAC_inv_loss_3')
##commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 3 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric cosine_origin --measure_all_similarities True --experiment_name PACS_40_clients_DAC_cosine_origin_3')
#ommands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 3 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric l2 --measure_all_similarities True --experiment_name PACS_40_clients_DAC_l2_3')
# testing for the previous four commands
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_DAC_cosine_initial_3')
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_DAC_inv_loss_3')
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_DAC_cosine_origin_3')
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_DAC_l2_3')

# Oracle and Random with inverse training loss
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 3 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange oracle --delusion -1 --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name PACS_40_clients_Random_3')
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 3 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange oracle --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name PACS_40_clients_Oracle_3')
# testing
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_Random_3')
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_Oracle_3')

# no communication
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --shift label --nbr_rounds 50 --nbr_clients 40 --n_data_train 400 --n_data_val 100 --seed 2 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --prior_update_rule softmax-variable-tau --similarity_metric inverse_training_loss --cosine_alpha 0 --tau 30 --client_information_exchange no_exchange --experiment_name PACS_40_clients_no_comm_2 --delusion 0 --NAEM_frequency 5 --T1 50 --CIFAR_ratio 0.4 --nbr_deluded_clients 0 --measure_all_similarities True --nbr_classes 10 --nbr_channels 3')
# testing
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_no_comm_2')

# all data
#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 4 --seed 2 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange no_exchange --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name PACS_40_clients_all_data_2')
# testing
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_all_data_2')

#commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric inverse_training_loss --measure_all_similarities True --experiment_name PACS_40_clients_DAC_inv_loss_1')
#commands.append('python3 test_PACS.py --experiment PACS_40_clients_DAC_inv_loss_1')

### LOOK HARD ONCE
# DAC with cosine similarity and inverse training loss, cosine origin and l2
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange look_hard_once --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name PACS_40_clients_LHO_cosine_initial_1')
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange look_hard_once --similarity_metric inverse_training_loss --measure_all_similarities True --experiment_name PACS_40_clients_LHO_inv_loss_1')
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange look_hard_once --similarity_metric cosine_origin --measure_all_similarities True --experiment_name PACS_40_clients_LHO_cosine_origin_1')
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange look_hard_once --similarity_metric l2 --measure_all_similarities True --experiment_name PACS_40_clients_LHO_l2_1')
# testing for the previous four commands
# commands.append('python3 test_PACS.py --experiment PACS_40_clients_LHO_cosine_initial_1')
# commands.append('python3 test_PACS.py --experiment PACS_40_clients_LHO_inv_loss_1')
# commands.append('python3 test_PACS.py --experiment PACS_40_clients_LHO_cosine_origin_1')
# commands.append('python3 test_PACS.py --experiment PACS_40_clients_LHO_l2_1')

### 10 local epochs for DAC with cosine similarity and inverse training loss ###
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric cosine_similarity --measure_all_similarities True --experiment_name PACS_40_clients_DAC_cosine_sim_10_epochs')
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric inverse_training_loss --measure_all_similarities True --experiment_name PACS_40_clients_DAC_inv_loss_10_epochs')

# Inverse training loss tau = 5, 1, 0.1 and 10 epochs, first with softmax-variable-tau, then with softmax
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric inverse_training_loss --measure_all_similarities True --tau 5 --experiment_name PACS_40_clients_DAC_inv_loss_10_epochs_tau5')

# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric inverse_training_loss --measure_all_similarities True --tau 0.01 --prior_update_rule softmax --experiment_name PACS_40_clients_DAC_inv_loss_10_epochs_tau0.01')

# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric inverse_training_loss --measure_all_similarities True --tau 0.1 --prior_update_rule softmax --experiment_name PACS_40_clients_DAC_inv_loss_10_epochs_tau0.1')

# same but for cosine similarity
# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric cosine_similarity --measure_all_similarities True --tau 5 --experiment_name PACS_40_clients_DAC_cosine_10_epochs_tau5')

# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric cosine_similarity --measure_all_similarities True --tau 0.01 --prior_update_rule softmax --experiment_name PACS_40_clients_DAC_cosine_10_epochs_tau0.01')

# commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 10 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric cosine_similarity --measure_all_similarities True --tau 0.1 --prior_update_rule softmax --experiment_name PACS_40_clients_DAC_cosine_10_epochs_tau0.1')

### unpretrained networks ###
commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 3 --lr 1e-03 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric inverse_training_loss --measure_all_similarities True --tau 30 --prior_update_rule softmax --experiment_name PACS_40_clients_DAC_inv_loss_3_epochs_lr_1e-03_unpretrained')

commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 3 --lr 1e-03 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange DAC --similarity_metric cosine_similarity --measure_all_similarities True --tau 30 --prior_update_rule softmax --experiment_name PACS_40_clients_DAC_cosine_3_epochs_lr_1e-03_unpretrained')

for command in commands:
    subprocess.run(command, shell=True)