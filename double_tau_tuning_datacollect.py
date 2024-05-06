from tau_tuning_experiment_datacollect import load_run_and_save

lr = 0.001
seed = 1

#### TRAININGWEIGHTS ####

### invloss ###
double_trainingweight_invloss_expnames = []
for tau in [5,10,30]:
    double_trainingweight_invloss_expnames.append('DOUBLE_invloss_seed_{}_tau_{}'.format(seed,tau))

### L2 ###
double_trainingweight_l2_expnames = []
for tau in [5,10,30]:
    double_trainingweight_l2_expnames.append('fashion_MNIST_DAC_l2_tau_{}_seed_{}'.format(tau,seed))

### cosine ###
double_trainingweight_cosine_expnames = []
for tau in [500,2000,5000]:
    double_trainingweight_cosine_expnames.append('DOUBLE_cosine_seed_{}_tau_{}'.format(seed,tau))

### cosine origin ###
double_trainingweight_cosine_origin_expnames = []
for tau in [500,2000,5000]:
    double_trainingweight_cosine_origin_expnames.append('DOUBLE_cosine_origin_seed_{}_tau_{}'.format(seed,tau))


#### PRIORWEIGHTS ####

### invloss ###
double_priorweight_invloss_expnames = []
for tau in [5,10,30]:
    double_priorweight_invloss_expnames.append('DOUBLE_priorweight_invloss_seed_{}_tau_{}'.format(seed,tau))

### L2 ###
double_priorweight_l2_expnames = []
for tau in [5,10,30]:
    double_priorweight_l2_expnames.append('DOUBLE_priorweight_l2_seed_{}_tau_{}'.format(seed,tau))

### cosine ###
double_priorweight_cosine_expnames = []
for tau in [500,2000,5000]:
    double_priorweight_cosine_expnames.append('DOUBLE_priorweight_cosine_seed_{}_tau_{}'.format(seed,tau))

### cosine origin ###
double_priorweight_cosine_origin_expnames = []
for tau in [500,2000,5000]:
    double_priorweight_cosine_origin_expnames.append('DOUBLE_priorweight_cosine_origin_seed_{}_tau_{}'.format(seed,tau))


double_trainingweight_expnames_collection = [double_trainingweight_invloss_expnames, double_trainingweight_l2_expnames, double_trainingweight_cosine_expnames, double_trainingweight_cosine_origin_expnames]
double_priorweight_expnames_collection = [double_priorweight_invloss_expnames, double_priorweight_l2_expnames, double_priorweight_cosine_expnames, double_priorweight_cosine_origin_expnames]

load_run_and_save(double_trainingweight_expnames_collection, 'double_trainingweight_results.pkl')
load_run_and_save(double_priorweight_expnames_collection, 'double_priorweight_results.pkl')

