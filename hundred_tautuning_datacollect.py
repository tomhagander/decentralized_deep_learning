from tau_tuning_experiment_datacollect import load_run_and_save
import numpy as np

hundred_pretrained_invloss_expnames = []
hundred_pretrained_l2_expnames = []
hundred_pretrained_cosine_expnames = []
hundred_pretrained_cosine_origin_expnames = []

invloss_taus = np.logspace(np.log10(1), np.log10(150), num=4)
l2_taus = np.logspace(np.log10(1), np.log10(150), num=4)
cosine_taus = np.logspace(np.log10(10), np.log10(300), num=4)
cosine_origin_taus = np.logspace(np.log10(10), np.log10(300), num=4)
cosine_taus = np.append(cosine_taus, [1, 5])
cosine_origin_taus = np.append(cosine_origin_taus, 5)
invloss_taus = np.append(invloss_taus, 0.5)

for tau in invloss_taus:
    hundred_pretrained_invloss_expnames.append('HUNDRED_pretrained_invloss_tau_{}'.format(tau))
for tau in l2_taus:
    hundred_pretrained_l2_expnames.append('HUNDRED_pretrained_l2_tau_{}'.format(tau))
for tau in cosine_taus:
    hundred_pretrained_cosine_expnames.append('HUNDRED_pretrained_cosine_tau_{}'.format(tau))
for tau in cosine_origin_taus:
    hundred_pretrained_cosine_origin_expnames.append('HUNDRED_pretrained_cosine_origin_tau_{}'.format(tau))


# nonpretrained
hundred_nonpretrained_invloss_expnames = []
hundred_nonpretrained_l2_expnames = []
hundred_nonpretrained_cosine_expnames = []
hundred_nonpretrained_cosine_origin_expnames = []

invloss_taus = np.logspace(np.log10(1), np.log10(150), num=4)
l2_taus = np.logspace(np.log10(1), np.log10(150), num=4)
cosine_taus = np.logspace(np.log10(10), np.log10(300), num=4)
cosine_origin_taus = np.logspace(np.log10(10), np.log10(300), num=4)
cosine_taus = np.append(cosine_taus, [1, 3, 5, 1000])
cosine_origin_taus = np.append(cosine_origin_taus, [5, 1000, 3000])

for tau in invloss_taus:
    hundred_nonpretrained_invloss_expnames.append('HUNDRED_nonpretrained_invloss_tau_{}'.format(tau))
for tau in l2_taus:
    hundred_nonpretrained_l2_expnames.append('HUNDRED_nonpretrained_l2_tau_{}'.format(tau))
for tau in cosine_taus:
    hundred_nonpretrained_cosine_expnames.append('HUNDRED_nonpretrained_cosine_tau_{}'.format(tau))
for tau in cosine_origin_taus:
    hundred_nonpretrained_cosine_origin_expnames.append('HUNDRED_nonpretrained_cosine_origin_tau_{}'.format(tau))


hundred_pretrained_expnames_collection = [hundred_pretrained_invloss_expnames, hundred_pretrained_l2_expnames, hundred_pretrained_cosine_expnames, hundred_pretrained_cosine_origin_expnames]
hundred_nonpretrained_expnames_collection = [hundred_nonpretrained_invloss_expnames, hundred_nonpretrained_l2_expnames, hundred_nonpretrained_cosine_expnames, hundred_nonpretrained_cosine_origin_expnames]


load_run_and_save(hundred_pretrained_expnames_collection, 'HUNDRED_pretrained_results.pkl')
load_run_and_save(hundred_nonpretrained_expnames_collection, 'HUNDRED_nonpretrained_results.pkl')