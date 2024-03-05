import subprocess

commands = []

commands.append('python3 local_trainer.py --experiment_name local_200 --n_data_train 200 --n_data_val 50')
commands.append('python3 local_trainer.py --experiment_name local_400 --n_data_train 400 --n_data_val 100')
commands.append('python3 local_trainer.py --experiment_name local_1000 --n_data_train 1000 --n_data_val 250')
commands.append('python3 local_trainer.py --experiment_name local_10000 --n_data_train 10000 --n_data_val 2500')

for command in commands:
    subprocess.run(command, shell=True)

