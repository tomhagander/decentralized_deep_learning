import subprocess

commands = []

lrs = [1e-4, 5e-4, 5e-5]
delusions = [-1, 0, 0.25, 0.75, 1]

###### oracle - change lr for all
for lr in lrs:
    for delusion in delusions:
        commands.append('python3 run_experiment.py --lr {} --client_information_exchange oracle --delusion {} --experiment_name big1_oracle_lr_{}_delusion_{}'.format(lr, delusion, lr, delusion))

lrs = [1e-4, 5e-4, 5e-5]
alphas = [0, 0.5, 1]
###### DAC - change lr for all
for lr in lrs:
    #inverse training loss
    commands.append('python3 run_experiment.py --lr {} --client_information_exchange DAC --similarity_metric inverse_training_loss --prior_update_rule softmax --tau 30 --experiment_name big1_DAC_lr_{}_inverse_training_loss_fixed_tau'.format(lr, lr))
    commands.append('python3 run_experiment.py --lr {} --client_information_exchange DAC --similarity_metric inverse_training_loss --prior_update_rule softmax-variable-tau --tau 30 --experiment_name big1_DAC_lr_{}_inverse_training_loss_variable_tau'.format(lr, lr))
    #cosine - change alpha for all - only do variable tau
    for alpha in alphas:
        commands.append('python3 run_experiment.py --lr {} --client_information_exchange DAC --similarity_metric cosine_similarity --cosine_alpha {} --prior_update_rule softmax-variable-tau --tau 30 --experiment_name big1_DAC_lr_{}_cosine_alpha_{}_variable_tau'.format(lr, alpha, lr, alpha))

lrs = [1e-4, 5e-4, 5e-5]
alphas = [0, 0.5, 1]
###### PANM - change lr for all
for lr in lrs:
    # inverse training loss
    commands.append('python3 run_experiment.py --lr {} --client_information_exchange PANM --similarity_metric inverse_training_loss --NAEM_frequency 2 --T1 80 --experiment_name big1_PANM_lr_{}_inverse_training_loss_NAEM_2'.format(lr, lr))
    commands.append('python3 run_experiment.py --lr {} --client_information_exchange PANM --similarity_metric inverse_training_loss --NAEM_frequency 10 --T1 80 --experiment_name big1_PANM_lr_{}_inverse_training_loss_NAEM_10'.format(lr, lr))
    # cosine - change alpha for all - only do quicker NAEM frequency
    for alpha in alphas:
        commands.append('python3 run_experiment.py --lr {} --client_information_exchange PANM --similarity_metric cosine_similarity --cosine_alpha {} --NAEM_frequency 2 --T1 80 --experiment_name big1_PANM_lr_{}_cosine_alpha_{}_NAEM_2'.format(lr, alpha, lr, alpha))

for command in commands:
    subprocess.run(command, shell=True)