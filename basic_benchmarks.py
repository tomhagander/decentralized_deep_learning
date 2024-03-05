import subprocess

commands = []

lr = 1e-5
commands.append('python3 run_experiment.py --lr {} --CIFAR_ratio 0.5 --nbr_clients 2 --nbr_local_epochs 1 --n_data_train 10000 --n_data_val 2000 --nbr_rounds 600 --client_information_exchange no_exchange --experiment_name local_benchmark_lr_{}_data_10000'.format(lr, lr))
commands.append('python3 run_experiment.py --lr {} --CIFAR_ratio 0.5 --nbr_clients 4 --nbr_local_epochs 3 --n_data_train 5000 --n_data_val 1000 --nbr_rounds 200 --client_information_exchange oracle --nbr_neighbors_sampled 1 --experiment_name 2_clients_benchmark_lr_{}__data_5000'.format(lr, lr))


for command in commands:
    subprocess.run(command, shell=True)