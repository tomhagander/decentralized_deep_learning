import subprocess

commands = []
commands.append('python3 run_experiment.py --dataset PACS --nbr_rounds 20 --nbr_clients 20 --lr 1e-4 --nbr_neighbors_sampled 2 --client_information_exchange oracle --experiment_name PACS_oracle')

for command in commands:
    subprocess.run(command, shell=True)