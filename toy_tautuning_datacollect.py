from tau_tuning_experiment_datacollect import load_run_and_save
import numpy as np

seed = 1
lrs = np.logspace(np.log10(0.008), np.log10(0.0003), num=4)
taus_trainingweight_invloss1 = np.logspace(np.log10(50000), np.log10(5000000), num=8)
taus_trainingweight_invloss2 = np.logspace(np.log10(10), np.log10(10000), num=8)
taus_trainingweight_invloss3 = np.logspace(np.log10(1), np.log(9), num=8)
# concat the two lists
taus_trainingweight_invloss = np.concatenate((taus_trainingweight_invloss1, taus_trainingweight_invloss2))
taus_trainingweight_invloss = np.concatenate((taus_trainingweight_invloss, taus_trainingweight_invloss3))

taus_trainingweight_l2 = np.logspace(np.log10(2), np.log10(100), num=8)
taus_trainingweight_cosine = np.logspace(np.log10(10), np.log10(1000), num=8)
taus_trainingweight_cosine_origin = np.logspace(np.log10(10), np.log10(1000), num=8)

taus_priorweight_invloss1 = np.logspace(np.log10(50000), np.log10(5000000), num=8)
taus_priorweight_invloss2 = np.logspace(np.log10(10), np.log10(10000), num=8)
# concat the two lists
taus_priorweight_invloss = np.concatenate((taus_priorweight_invloss1, taus_priorweight_invloss2))

taus_priorweight_l2 = np.logspace(np.log10(2), np.log10(500), num=8)
taus_priorweight_cosine = np.logspace(np.log10(50), np.log10(5000), num=8)
taus_priorweight_cosine_origin = np.logspace(np.log10(50), np.log10(5000), num=8)

####

toy_invloss_expnames_lr1 = []
toy_l2_expnames_lr1 = []
toy_cosine_expnames_lr1 = []
toy_origin_expnames_lr1 = []
toy_priorweight_invloss_expnames_lr1 = []
toy_priorweight_l2_expnames_lr1 = []
toy_priorweight_cosine_expnames_lr1 = []
toy_priorweight_origin_expnames_lr1 = []

toy_invloss_expnames_lr2 = []
toy_l2_expnames_lr2 = []
toy_cosine_expnames_lr2 = []
toy_origin_expnames_lr2 = []
toy_priorweight_invloss_expnames_lr2 = []
toy_priorweight_l2_expnames_lr2 = []
toy_priorweight_cosine_expnames_lr2 = []
toy_priorweight_origin_expnames_lr2 = []

toy_invloss_expnames_lr3 = []
toy_l2_expnames_lr3 = []
toy_cosine_expnames_lr3 = []
toy_origin_expnames_lr3 = []
toy_priorweight_invloss_expnames_lr3 = []
toy_priorweight_l2_expnames_lr3 = []
toy_priorweight_cosine_expnames_lr3 = []
toy_priorweight_origin_expnames_lr3 = []

toy_invloss_expnames_lr4 = []
toy_l2_expnames_lr4 = []
toy_cosine_expnames_lr4 = []
toy_origin_expnames_lr4 = []
toy_priorweight_invloss_expnames_lr4 = []
toy_priorweight_l2_expnames_lr4 = []
toy_priorweight_cosine_expnames_lr4 = []
toy_priorweight_origin_expnames_lr4 = []

lr = lrs[0]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_invloss_expnames_lr1.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_l2_expnames_lr1.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_cosine_expnames_lr1.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_origin_expnames_lr1.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_invloss_expnames_lr1.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_l2_expnames_lr1.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_cosine_expnames_lr1.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_origin_expnames_lr1.append(expname)

lr = lrs[1]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_invloss_expnames_lr2.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_l2_expnames_lr2.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_cosine_expnames_lr2.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_origin_expnames_lr2.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_invloss_expnames_lr2.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_l2_expnames_lr2.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_cosine_expnames_lr2.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_origin_expnames_lr2.append(expname)

    

lr = lrs[2]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_invloss_expnames_lr3.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_l2_expnames_lr3.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_cosine_expnames_lr3.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_origin_expnames_lr3.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_invloss_expnames_lr3.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_l2_expnames_lr3.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_cosine_expnames_lr3.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_origin_expnames_lr3.append(expname)

lr = lrs[3]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_invloss_expnames_lr4.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_l2_expnames_lr4.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_cosine_expnames_lr4.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_origin_expnames_lr4.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_invloss_expnames_lr4.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_l2_expnames_lr4.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_cosine_expnames_lr4.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_origin_expnames_lr4.append(expname)

#### MINMAX
taus_trainingweight_invloss1 = np.logspace(np.log10(50000), np.log10(5000000), num=8)
taus_trainingweight_invloss2 = np.logspace(np.log10(1), np.log10(10000), num=8)
# concat the arrays
taus_trainingweight_invloss = np.concatenate((taus_trainingweight_invloss1, taus_trainingweight_invloss2))
taus_trainingweight_l2 = np.logspace(np.log10(2), np.log10(100), num=8)
taus_trainingweight_cosine = np.logspace(np.log10(10), np.log10(1000), num=8)
taus_trainingweight_cosine_origin = np.logspace(np.log10(10), np.log10(1000), num=8)
taus_priorweight_invloss1 = np.logspace(np.log10(50000), np.log10(5000000), num=8)
taus_priorweight_invloss2 = np.logspace(np.log10(10), np.log10(10000), num=8)
# concat the arrays
taus_priorweight_invloss = np.concatenate((taus_priorweight_invloss1, taus_priorweight_invloss2))
taus_priorweight_l2 = np.logspace(np.log10(2), np.log10(500), num=8)
taus_priorweight_cosine = np.logspace(np.log10(50), np.log10(5000), num=8)
taus_priorweight_cosine_origin = np.logspace(np.log10(50), np.log10(5000), num=8)

toy_trainingweight_minmax_invloss_expnames_lr1 = []
toy_trainingweight_minmax_l2_expnames_lr1 = []
toy_trainingweight_minmax_cosine_expnames_lr1 = []
toy_trainingweight_minmax_cosine_origin_expnames_lr1 = []

toy_trainingweight_minmax_invloss_expnames_lr2 = []
toy_trainingweight_minmax_l2_expnames_lr2 = []
toy_trainingweight_minmax_cosine_expnames_lr2 = []
toy_trainingweight_minmax_cosine_origin_expnames_lr2 = []

toy_trainingweight_minmax_invloss_expnames_lr3 = []
toy_trainingweight_minmax_l2_expnames_lr3 = []
toy_trainingweight_minmax_cosine_expnames_lr3 = []
toy_trainingweight_minmax_cosine_origin_expnames_lr3 = []

toy_trainingweight_minmax_invloss_expnames_lr4 = []
toy_trainingweight_minmax_l2_expnames_lr4 = []
toy_trainingweight_minmax_cosine_expnames_lr4 = []
toy_trainingweight_minmax_cosine_origin_expnames_lr4 = []

toy_priorweight_minmax_invloss_expnames_lr1 = []
toy_priorweight_minmax_l2_expnames_lr1 = []
toy_priorweight_minmax_cosine_expnames_lr1 = []
toy_priorweight_minmax_cosine_origin_expnames_lr1 = []

toy_priorweight_minmax_invloss_expnames_lr2 = []
toy_priorweight_minmax_l2_expnames_lr2 = []
toy_priorweight_minmax_cosine_expnames_lr2 = []
toy_priorweight_minmax_cosine_origin_expnames_lr2 = []

toy_priorweight_minmax_invloss_expnames_lr3 = []
toy_priorweight_minmax_l2_expnames_lr3 = []
toy_priorweight_minmax_cosine_expnames_lr3 = []
toy_priorweight_minmax_cosine_origin_expnames_lr3 = []

toy_priorweight_minmax_invloss_expnames_lr4 = []
toy_priorweight_minmax_l2_expnames_lr4 = []
toy_priorweight_minmax_cosine_expnames_lr4 = []
toy_priorweight_minmax_cosine_origin_expnames_lr4 = []

lr = lrs[0]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_minmax_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_invloss_expnames_lr1.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_minmax_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_l2_expnames_lr1.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_expnames_lr1.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_origin_expnames_lr1.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_minmax_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_invloss_expnames_lr1.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_minmax_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_l2_expnames_lr1.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_expnames_lr1.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_origin_expnames_lr1.append(expname)

lr = lrs[1]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_minmax_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_invloss_expnames_lr2.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_minmax_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_l2_expnames_lr2.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_expnames_lr2.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_origin_expnames_lr2.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_minmax_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_invloss_expnames_lr2.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_minmax_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_l2_expnames_lr2.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_expnames_lr2.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_origin_expnames_lr2.append(expname)

lr = lrs[2]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_minmax_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_invloss_expnames_lr3.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_minmax_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_l2_expnames_lr3.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_expnames_lr3.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_origin_expnames_lr3.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_minmax_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_invloss_expnames_lr3.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_minmax_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_l2_expnames_lr3.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_expnames_lr3.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_origin_expnames_lr3.append(expname)

lr = lrs[3]
for tau in taus_trainingweight_invloss:
    expname = 'TOY_DAC_minmax_invloss_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_invloss_expnames_lr4.append(expname)
for tau in taus_trainingweight_l2:
    expname = 'TOY_DAC_minmax_l2_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_l2_expnames_lr4.append(expname)
for tau in taus_trainingweight_cosine:
    expname = 'TOY_DAC_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_expnames_lr4.append(expname)
for tau in taus_trainingweight_cosine_origin:
    expname = 'TOY_DAC_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_trainingweight_minmax_cosine_origin_expnames_lr4.append(expname)
for tau in taus_priorweight_invloss:
    expname = 'TOY_DAC_priorweight_minmax_invloss_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_invloss_expnames_lr4.append(expname)
for tau in taus_priorweight_l2:
    expname = 'TOY_DAC_priorweight_minmax_l2_prior_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_l2_expnames_lr4.append(expname)
for tau in taus_priorweight_cosine:
    expname = 'TOY_DAC_priorweight_minmax_cosine_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_expnames_lr4.append(expname)
for tau in taus_priorweight_cosine_origin:
    expname = 'TOY_DAC_priorweight_minmax_cosine_origin_lr_{}_tau_{}_seed_1'.format(lr, tau)
    toy_priorweight_minmax_cosine_origin_expnames_lr4.append(expname)


toy_trainingweight_expnames_collection_lr1 = [toy_invloss_expnames_lr1, toy_l2_expnames_lr1, toy_cosine_expnames_lr1, toy_origin_expnames_lr1]
toy_trainingweight_expnames_collection_lr2 = [toy_invloss_expnames_lr2, toy_l2_expnames_lr2, toy_cosine_expnames_lr2, toy_origin_expnames_lr2]
toy_trainingweight_expnames_collection_lr3 = [toy_invloss_expnames_lr3, toy_l2_expnames_lr3, toy_cosine_expnames_lr3, toy_origin_expnames_lr3]
toy_trainingweight_expnames_collection_lr4 = [toy_invloss_expnames_lr4, toy_l2_expnames_lr4, toy_cosine_expnames_lr4, toy_origin_expnames_lr4]
toy_priorweight_expnames_collection_lr1 = [toy_priorweight_invloss_expnames_lr1, toy_priorweight_l2_expnames_lr1, toy_priorweight_cosine_expnames_lr1, toy_priorweight_origin_expnames_lr1]
toy_priorweight_expnames_collection_lr2 = [toy_priorweight_invloss_expnames_lr2, toy_priorweight_l2_expnames_lr2, toy_priorweight_cosine_expnames_lr2, toy_priorweight_origin_expnames_lr2]
toy_priorweight_expnames_collection_lr3 = [toy_priorweight_invloss_expnames_lr3, toy_priorweight_l2_expnames_lr3, toy_priorweight_cosine_expnames_lr3, toy_priorweight_origin_expnames_lr3]
toy_priorweight_expnames_collection_lr4 = [toy_priorweight_invloss_expnames_lr4, toy_priorweight_l2_expnames_lr4, toy_priorweight_cosine_expnames_lr4, toy_priorweight_origin_expnames_lr4]
toy_trainingweight_minmax_expnames_collection_lr1 = [toy_trainingweight_minmax_invloss_expnames_lr1, toy_trainingweight_minmax_l2_expnames_lr1, toy_trainingweight_minmax_cosine_expnames_lr1, toy_trainingweight_minmax_cosine_origin_expnames_lr1]
toy_trainingweight_minmax_expnames_collection_lr2 = [toy_trainingweight_minmax_invloss_expnames_lr2, toy_trainingweight_minmax_l2_expnames_lr2, toy_trainingweight_minmax_cosine_expnames_lr2, toy_trainingweight_minmax_cosine_origin_expnames_lr2]
toy_trainingweight_minmax_expnames_collection_lr3 = [toy_trainingweight_minmax_invloss_expnames_lr3, toy_trainingweight_minmax_l2_expnames_lr3, toy_trainingweight_minmax_cosine_expnames_lr3, toy_trainingweight_minmax_cosine_origin_expnames_lr3]
toy_trainingweight_minmax_expnames_collection_lr4 = [toy_trainingweight_minmax_invloss_expnames_lr4, toy_trainingweight_minmax_l2_expnames_lr4, toy_trainingweight_minmax_cosine_expnames_lr4, toy_trainingweight_minmax_cosine_origin_expnames_lr4]
toy_priorweight_minmax_expnames_collection_lr1 = [toy_priorweight_minmax_invloss_expnames_lr1, toy_priorweight_minmax_l2_expnames_lr1, toy_priorweight_minmax_cosine_expnames_lr1, toy_priorweight_minmax_cosine_origin_expnames_lr1]
toy_priorweight_minmax_expnames_collection_lr2 = [toy_priorweight_minmax_invloss_expnames_lr2, toy_priorweight_minmax_l2_expnames_lr2, toy_priorweight_minmax_cosine_expnames_lr2, toy_priorweight_minmax_cosine_origin_expnames_lr2]
toy_priorweight_minmax_expnames_collection_lr3 = [toy_priorweight_minmax_invloss_expnames_lr3, toy_priorweight_minmax_l2_expnames_lr3, toy_priorweight_minmax_cosine_expnames_lr3, toy_priorweight_minmax_cosine_origin_expnames_lr3]
toy_priorweight_minmax_expnames_collection_lr4 = [toy_priorweight_minmax_invloss_expnames_lr4, toy_priorweight_minmax_l2_expnames_lr4, toy_priorweight_minmax_cosine_expnames_lr4, toy_priorweight_minmax_cosine_origin_expnames_lr4]

load_run_and_save(toy_trainingweight_expnames_collection_lr1, 'toy_trainingweight_results_lr1.pkl', toy=True)
load_run_and_save(toy_trainingweight_expnames_collection_lr2, 'toy_trainingweight_results_lr2.pkl', toy=True)
load_run_and_save(toy_trainingweight_expnames_collection_lr3, 'toy_trainingweight_results_lr3.pkl', toy=True)
load_run_and_save(toy_trainingweight_expnames_collection_lr4, 'toy_trainingweight_results_lr4.pkl', toy=True)
load_run_and_save(toy_priorweight_expnames_collection_lr1, 'toy_priorweight_results_lr1.pkl', toy=True)
load_run_and_save(toy_priorweight_expnames_collection_lr2, 'toy_priorweight_results_lr2.pkl', toy=True)
load_run_and_save(toy_priorweight_expnames_collection_lr3, 'toy_priorweight_results_lr3.pkl', toy=True)
load_run_and_save(toy_priorweight_expnames_collection_lr4, 'toy_priorweight_results_lr4.pkl', toy=True)
load_run_and_save(toy_trainingweight_minmax_expnames_collection_lr1, 'toy_trainingweight_minmax_results_lr1.pkl', toy=True)
load_run_and_save(toy_trainingweight_minmax_expnames_collection_lr2, 'toy_trainingweight_minmax_results_lr2.pkl', toy=True)
load_run_and_save(toy_trainingweight_minmax_expnames_collection_lr3, 'toy_trainingweight_minmax_results_lr3.pkl', toy=True)
load_run_and_save(toy_trainingweight_minmax_expnames_collection_lr4, 'toy_trainingweight_minmax_results_lr4.pkl', toy=True)
load_run_and_save(toy_priorweight_minmax_expnames_collection_lr1, 'toy_priorweight_minmax_results_lr1.pkl', toy=True)
load_run_and_save(toy_priorweight_minmax_expnames_collection_lr2, 'toy_priorweight_minmax_results_lr2.pkl', toy=True)
load_run_and_save(toy_priorweight_minmax_expnames_collection_lr3, 'toy_priorweight_minmax_results_lr3.pkl', toy=True)
load_run_and_save(toy_priorweight_minmax_expnames_collection_lr4, 'toy_priorweight_minmax_results_lr4.pkl', toy=True)

