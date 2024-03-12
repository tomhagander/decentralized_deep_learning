import subprocess

commands = []

nbr_clients = [4,8,10,20,40]

delusions = [-1, 0, 0.25, 0.75, 1]

###### oracle - change lr for all
for c in nbr_clients:
    commands.append('run_experiment.py --lr 1e-5 --CIFAR_ratio 0.51 --nbr_clients {} --nbr_local_epochs 3 --n_data_train {} --stopping_rounds 30 --n_data_val {} --nbr_rounds 200 --client_information_exchange oracle --nbr_neighbors_sampled {} --experiment_name {}_clients_benchmark_oracle_data_per_client_{}'.format(c, 10000/(c/2), 2000, c/2, c, 10000))


#run_experiment.py --lr 1e-5 --CIFAR_ratio 0.55 --nbr_clients 4 --nbr_local_epochs 3 --n_data_train 5000 --n_data_val 1000 --nbr_rounds 200 --client_information_exchange oracle --nbr_neighbors_sampled 1 --experiment_name 2_clients_benchmark_lr_1e-5__data_5000


for command in commands:
    subprocess.run(command, shell=True)