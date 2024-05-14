from tau_tuning_experiment_datacollect import load_run_and_save

lr = 0.0001
taus = [1, 5, 10, 30, 100, 300, 1000, 3000]

# trainingweight

# invloss
double_MLP_trainingweight_invloss_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_invloss_trainingweight_tau_{}'.format(tau)
    double_MLP_trainingweight_invloss_expnames.append(expname)

# L2
double_MLP_trainingweight_l2_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_l2_trainingweight_tau_{}'.format(tau)
    double_MLP_trainingweight_l2_expnames.append(expname)

# cosine
double_MLP_trainingweight_cosine_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_cosine_trainingweight_tau_{}'.format(tau)
    double_MLP_trainingweight_cosine_expnames.append(expname)

# cosine origin
double_MLP_trainingweight_cosine_origin_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_cosine_origin_trainingweight_tau_{}'.format(tau)
    double_MLP_trainingweight_cosine_origin_expnames.append(expname)

# priorweight

# invloss
double_MLP_priorweight_invloss_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_invloss_priorweight_tau_{}'.format(tau)
    double_MLP_priorweight_invloss_expnames.append(expname)

# L2
double_MLP_priorweight_l2_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_l2_priorweight_tau_{}'.format(tau)
    double_MLP_priorweight_l2_expnames.append(expname)

# cosine
double_MLP_priorweight_cosine_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_cosine_priorweight_tau_{}'.format(tau)
    double_MLP_priorweight_cosine_expnames.append(expname)

# cosine origin
double_MLP_priorweight_cosine_origin_expnames = []
for tau in taus:
    expname = 'DOUBLE_MLP_cosine_origin_priorweight_tau_{}'.format(tau)
    double_MLP_priorweight_cosine_origin_expnames.append(expname)

double_MLP_trainingweight_expnames_collection = [double_MLP_trainingweight_invloss_expnames, double_MLP_trainingweight_l2_expnames, double_MLP_trainingweight_cosine_expnames, double_MLP_trainingweight_cosine_origin_expnames]
double_MLP_priorweight_expnames_collection = [double_MLP_priorweight_invloss_expnames, double_MLP_priorweight_l2_expnames, double_MLP_priorweight_cosine_expnames, double_MLP_priorweight_cosine_origin_expnames]

load_run_and_save(double_MLP_trainingweight_expnames_collection, 'double_MLP_trainingweight_results.pkl')
load_run_and_save(double_MLP_priorweight_expnames_collection, 'double_MLP_priorweight_results.pkl')
