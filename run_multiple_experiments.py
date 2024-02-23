import subprocess


'''
# List of parameters
parameters = [
    ['--nbr_clients', '100', 
     '--nbr_rounds', '200',
     '--n_data_train', '400', 
     '--n_data_val', '100', 
     '--seed', '1', 
     '--batch_size', '8', 
     '--nbr_local_epochs', '3', 
     '--lr', '3e-5', 
     '--nbr_classes', '10', 
     '--nbr_channels', '3', 
     '--stopping_rounds', '10', 
     '--nbr_neighbors_sampled', '5', 
     '--prior_update_rule', 'softmax', 
     '--similarity_metric', 'inverse_training_loss', 
     '--tau', '30', 
     '--experiment_name', 'replication_DACvar_1'],
]

# Iterate over the parameters
for param in parameters:
    # Build the command
    command = ['python3', 'run_experiment.py',
            param[0], param[1], 
            param[2], param[3], 
            param[4], param[5], 
            param[6], param[7], 
            param[8], param[9], 
            param[10], param[11], 
            param[12], param[13], 
            param[14], param[15], 
            param[16], param[17], 
            param[18], param[19],
            param[20], param[21],
            param[22], param[23],
            param[24], param[25],
            param[26], param[27],
            param[28], param[29],
            ]
    
    # Run the command
    subprocess.run(command)
'''

command  = 'python3 run_experiment.py --nbr_clients 10 --nbr_rounds 1 --n_data_train 400 --n_data_val 100 --seed 1 --batch_size 8 --nbr_local_epochs 3 --lr 3e-5 --nbr_classes 10 --nbr_channels 3 --stopping_rounds 10 --nbr_neighbors_sampled 5 --prior_update_rule softmax --similarity_metric inverse_training_loss --tau 30 --experiment_name dev2'
subprocess.run(command, shell=True)