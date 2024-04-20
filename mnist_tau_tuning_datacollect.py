from tau_tuning_experiment_datacollect import load_run_and_save

mnist_trainingweight_invloss_expnames = ['fashion_MNIST_DAC_invloss_tau_5_seed_1',
                                         'fashion_MNIST_DAC_invloss_tau_10_seed_1',
                                         'fashion_MNIST_DAC_invloss_tau_30_seed_1',]

mnist_trainingweight_l2_expnames = ['fashion_MNIST_DAC_l2_tau_1_seed_1',
                                    'fashion_MNIST_DAC_l2_tau_5_seed_1',
                                    'fashion_MNIST_DAC_l2_tau_10_seed_1',
                                    'fashion_MNIST_DAC_l2_tau_30_seed_1',
                                    'fashion_MNIST_DAC_l2_tau_80_seed_1',]

mnist_trainingweight_cosine_expnames = ['fashion_MNIST_DAC_cosine_tau_100_seed_1',
                                        'fashion_MNIST_DAC_cosine_tau_200_seed_1',
                                        'fashion_MNIST_DAC_cosine_tau_300_seed_1',
                                        'fashion_MNIST_DAC_cosine_tau_500_seed_1',
                                        'fashion_MNIST_DAC_cosine_tau_2000_seed_1']

mnist_trainingweight_origin_expnames = ['fashion_MNIST_DAC_cosine_origin_tau_50_seed_1',
                                        'fashion_MNIST_DAC_cosine_origin_tau_100_seed_1',
                                        'fashion_MNIST_DAC_cosine_origin_tau_200_seed_1',
                                        'fashion_MNIST_DAC_cosine_origin_tau_300_seed_1',
                                        'fashion_MNIST_DAC_cosine_origin_tau_500_seed_1',
                                        'fashion_MNIST_DAC_cosine_origin_tau_2000_seed_1']

mnist_priorweight_invloss_expnames = ['fashion_MNIST_DAC_priorweight_invloss_tau_1_seed_1',
                                      'fashion_MNIST_DAC_priorweight_invloss_tau_5_seed_1',
                                      'fashion_MNIST_DAC_priorweight_invloss_tau_10_seed_1',
                                      'fashion_MNIST_DAC_priorweight_invloss_tau_30_seed_1',]

mnist_priorweight_l2_expnames = ['fashion_MNIST_DAC_priorweight_l2_tau_1_seed_1',
                                 'fashion_MNIST_DAC_priorweight_l2_tau_5_seed_1',
                                 'fashion_MNIST_DAC_priorweight_l2_tau_10_seed_1',
                                 'fashion_MNIST_DAC_priorweight_l2_tau_30_seed_1',
                                 'fashion_MNIST_DAC_priorweight_l2_tau_80_seed_1',]

mnist_priorweight_cosine_expnames = ['fashion_MNIST_DAC_priorweight_cosine_tau_100_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_tau_200_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_tau_300_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_tau_500_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_tau_2000_seed_1']

mnist_priorweight_origin_expnames = ['fashion_MNIST_DAC_priorweight_cosine_origin_tau_100_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_origin_tau_200_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_origin_tau_300_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_origin_tau_500_seed_1',
                                        'fashion_MNIST_DAC_priorweight_cosine_origin_tau_2000_seed_1']

mnist_trainingweight_expnames_collection = [mnist_trainingweight_invloss_expnames, mnist_trainingweight_l2_expnames, mnist_trainingweight_cosine_expnames, mnist_trainingweight_origin_expnames]
mnist_priorweight_expnames_collection = [mnist_priorweight_invloss_expnames, mnist_priorweight_l2_expnames, mnist_priorweight_cosine_expnames, mnist_priorweight_origin_expnames]

load_run_and_save(mnist_trainingweight_expnames_collection, 'mnist_trainingweight_results.pkl')
load_run_and_save(mnist_priorweight_expnames_collection, 'mnist_priorweight_results.pkl')