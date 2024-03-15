import subprocess

commands = []

# tune learning rate
lrs = [0.00025, 0.000075, 0.00005, 0.000025]
bss = [8, 32, 64, 128]
for i, lr in enumerate(lrs):
    bs = bss[i]
    commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 42 --batch_size {} --nbr_local_epochs 1 --lr {} --stopping_rounds 30 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --cosine_alpha 0.5 --tau 1 --client_information_exchange oracle --experiment_name PACS_findlr_{} --delusion 0.0 --NAEM_frequency 5 --T1 50 --nbr_classes 10 --nbr_channels 3'.format(bs, lr, lr))

# run local training
lrs = [0.00025, 0.000075, 0.00005, 0.0001]
for lr in lrs:
    commands.append('python3 run_experiment.py --gpu 0 --dataset PACS --nbr_rounds 50 --nbr_clients 100 --n_data_train 400 --n_data_val 100 --seed 42 --batch_size 128 --nbr_local_epochs 1 --lr {} --stopping_rounds 30 --client_information_exchange no_exchange --experiment_name PACS_local_lr_{} --nbr_classes 10 --nbr_channels 3'.format(lr, lr))

for command in commands:
    subprocess.run(command, shell=True)