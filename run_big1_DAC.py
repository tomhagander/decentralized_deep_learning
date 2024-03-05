import subprocess

commands = []

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

for command in commands:
    subprocess.run(command, shell=True)