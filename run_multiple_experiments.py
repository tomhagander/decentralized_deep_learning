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
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange DAC --tau 30 --similarity_metric inverse_training_loss --experiment_name RECREATE_DAC_INV_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with DAC, variable tau, tau 30, cosine similarity, alpha 0.5

'''
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange DAC --tau 30 --similarity_metric cosine_similarity --cosine_alpha 0.5 --experiment_name RECREATE_DAC_COS_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with PANM, T1 100, NAEM frequency 2, inverse_training_loss
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange PANM --T1 100 --NAEM_frequency 2 --similarity_metric inverse_training_loss --experiment_name RECREATE_PANM_INV_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))
# now with PANM, T1 100, NAEM frequency 2, cosine similarity, alpha 0.5
commands.append('python3 run_experiment.py --gpu 0 --dataset cifar10 --shift PANM_swap4 --nbr_rounds 270 --nbr_clients 100 --n_data_train 200 --n_data_val 100 --seed 1 --batch_size 128 --nbr_local_epochs 3 --lr {} --stopping_rounds 50 --nbr_neighbors_sampled 5 --client_information_exchange PANM --T1 100 --NAEM_frequency 2 --similarity_metric cosine_similarity --cosine_alpha 0.5 --experiment_name RECREATE_PANM_COS_SWAP4_lr_{} --CIFAR_ratio 0.25 --nbr_classes 10 --nbr_channels 3'.format(lr, lr))


for command in commands:
    subprocess.run(command, shell=True)