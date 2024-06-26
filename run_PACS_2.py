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
commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 40 --seed 1 --batch_size 32 --nbr_local_epochs 1 --lr 7.5e-05 --stopping_rounds 30 --nbr_neighbors_sampled 2 --client_information_exchange no_exchange --similarity_metric cosine_origin --measure_all_similarities True --experiment_name PACS_40_clients_no_comm_4')
commands.append('python3 test_PACS.py --experiment PACS_40_clients_no_comm_4')


for command in commands:
    subprocess.run(command, shell=True)