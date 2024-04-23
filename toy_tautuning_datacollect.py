from tau_tuning_experiment_datacollect import load_run_and_save

toy_invloss_expnames = ['TOY_DAC_invloss_tau_1_seed_1',
                        'TOY_DAC_invloss_tau_10_seed_1',
                        'TOY_DAC_invloss_tau_100_seed_1',
                        'TOY_DAC_invloss_tau_1000_seed_1',
                        'TOY_DAC_invloss_tau_10000_seed_1',]

toy_l2_expnames = ['TOY_DAC_l2_tau_1_seed_1',
                     'TOY_DAC_l2_tau_10_seed_1',
                     'TOY_DAC_l2_tau_100_seed_1',
                     'TOY_DAC_l2_tau_1000_seed_1',
                     'TOY_DAC_l2_tau_10000_seed_1',]

toy_cosine_expnames = ['TOY_DAC_cosine_tau_1_seed_1',
                        'TOY_DAC_cosine_tau_10_seed_1',
                        'TOY_DAC_cosine_tau_100_seed_1',
                        'TOY_DAC_cosine_tau_1000_seed_1',
                        'TOY_DAC_cosine_tau_10000_seed_1',]

toy_origin_expnames = ['TOY_DAC_cosine_origin_tau_1_seed_1',
                        'TOY_DAC_cosine_origin_tau_10_seed_1',
                        'TOY_DAC_cosine_origin_tau_100_seed_1',
                        'TOY_DAC_cosine_origin_tau_1000_seed_1',
                        'TOY_DAC_cosine_origin_tau_10000_seed_1',]

toy_priorweight_invloss_expnames = ['TOY_DAC_priorweight_invloss_tau_1_seed_1',
                                    'TOY_DAC_priorweight_invloss_tau_10_seed_1',
                                    'TOY_DAC_priorweight_invloss_tau_100_seed_1',
                                    'TOY_DAC_priorweight_invloss_tau_1000_seed_1',
                                    'TOY_DAC_priorweight_invloss_tau_10000_seed_1',]

toy_priorweight_l2_expnames = ['TOY_DAC_priorweight_l2_tau_1_seed_1',
                                    'TOY_DAC_priorweight_l2_tau_10_seed_1',
                                    'TOY_DAC_priorweight_l2_tau_100_seed_1',
                                    'TOY_DAC_priorweight_l2_tau_1000_seed_1',
                                    'TOY_DAC_priorweight_l2_tau_10000_seed_1',]

toy_priorweight_cosine_expnames = ['TOY_DAC_priorweight_cosine_tau_1_seed_1',
                                    'TOY_DAC_priorweight_cosine_tau_10_seed_1',
                                    'TOY_DAC_priorweight_cosine_tau_100_seed_1',
                                    'TOY_DAC_priorweight_cosine_tau_1000_seed_1',
                                    'TOY_DAC_priorweight_cosine_tau_10000_seed_1',]

toy_priorweight_origin_expnames = ['TOY_DAC_priorweight_cosine_origin_tau_1_seed_1',
                                    'TOY_DAC_priorweight_cosine_origin_tau_10_seed_1',
                                    'TOY_DAC_priorweight_cosine_origin_tau_100_seed_1',
                                    'TOY_DAC_priorweight_cosine_origin_tau_1000_seed_1',
                                    'TOY_DAC_priorweight_cosine_origin_tau_10000_seed_1',]


toy_trainingweight_expnames_collection = [toy_invloss_expnames, toy_l2_expnames, toy_cosine_expnames, toy_origin_expnames]
toy_priorweight_expnames_collection = [toy_priorweight_invloss_expnames, toy_priorweight_l2_expnames, toy_priorweight_cosine_expnames, toy_priorweight_origin_expnames]

load_run_and_save(toy_trainingweight_expnames_collection, 'toy_trainingweight_results.pkl')
load_run_and_save(toy_priorweight_expnames_collection, 'toy_priorweight_results.pkl')