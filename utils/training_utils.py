import numpy as np
from scipy.stats import norm
from copy import deepcopy
import copy
import torch

def FedAvg(w,alpha):
    w_avg = copy.deepcopy(w[0])
    n_clients = len(w)
    alpha = alpha/np.sum(alpha)
    for l in w_avg.keys():
        w_avg[l] = w_avg[l].float() - w_avg[l]

    for l, layer in enumerate(w_avg.keys()): #for each layer
        for k in range(0,n_clients): #for each client
            w_avg[layer] += alpha[k]*w[k][layer]
    return w_avg

def tau_function(x,a,b):
    tau = 2*a/(1+np.exp(-b*x)) - a +1
    return tau

def _find_tau(similarities, entropy, lower_bound, upper_bound):
    # calculate the middle point
    tau = (lower_bound + upper_bound) / 2
    # calculate the entropy for the given tau
    # priors given tau: 
    priors = np.exp(similarities*tau - max(similarities*tau))
    priors = priors / np.sum(priors)
    entropy_tau = -np.sum([(p)*np.log2(p) for p in priors if p > 0])
    # if the entropy is close enough to the desired entropy, return the tau

    print('tau: {}, entropy_tau: {}, entropy: {}'.format(tau, entropy_tau, entropy))

    if np.abs(entropy_tau - entropy) < 0.1:
        return tau
    # if the entropy is too high, the tau is too low, and vice versa
    elif entropy_tau > entropy:
        return _find_tau(similarities, entropy, tau, upper_bound)
    else:
        return _find_tau(similarities, entropy, lower_bound, tau)

def find_tau(similarities, entropy):
    # call function recursively to minimize the interval of tau

    print('Starting to find tau for new client')
    print('Similarities: {}'.format(similarities))

    lower_bound = 0
    upper_bound = 10000
    tau = _find_tau(similarities, entropy, lower_bound, upper_bound)
    return tau

# trains all clients locally, one by one, for the set number of local epochs
def train_clients_locally(clients, nbr_local_epochs, verbose=False):
    if verbose:
        print('Starting local training')
    for client in clients:
        if not client.early_stopping.is_stopped():
            if verbose:
                print('Training client {}'.format(client.idx))
            client.train(nbr_local_epochs)
        else:
            if verbose:
                print('Client {} stopped early'.format(client.idx))
    return clients

# clients exchange information with each other asynchronously in a random order
# the function is the same for all similarity metrics and how those are distributed in the network
def client_information_exchange_DAC(clients, parameters, verbose=False, round=0):
    '''
    parameters:
    n_sampled: number of neighbors sampled
    prior_update_rule: how to update priors
    similarity_metric: how to measure similarity between clients
    '''

    inv_epsilon = 1e-6

    if verbose:
        print('Starting information exchange round {}'.format(round))

    # check if all clients have stopped early
    all_stopped = True
    for client in clients:
        if not client.early_stopping.is_stopped():
            all_stopped = False
            break
    
    if all_stopped:
        if verbose:
            print('All clients have stopped early, stopping information exchange')
        return clients

    # randomize order of asynchronous communication
    idxs = np.arange(len(clients))
    np.random.shuffle(idxs)
    for i in idxs:
        if verbose:
            if clients[i].early_stopping.is_stopped():
                print('Client {} has stopped early'.format(i))
        if (not clients[i].early_stopping.is_stopped()):
            
            #### sample neighbors
            # question: what about variable number of neighbors sampled? TODO
            neighbor_indices_sampled = np.random.choice(len(clients), 
                                                        size=parameters['nbr_neighbors_sampled'], 
                                                        replace=False, 
                                                        p=clients[i].priors)

            #### aggregate information from chosen neighbors
            # potentially different model aggregation weightings here
            neighbor_weights = []
            train_losses_ij = []
            train_set_sizes = []
            for j in neighbor_indices_sampled:
                # validate on neighbor model, get loss and accuracy
                neighbor_model = clients[j].local_model
                train_set_size = len(clients[j].train_set)
                train_loss_ij, train_acc_ij = clients[i].validate(neighbor_model, train_set=True)
                # save stuff
                train_losses_ij.append(train_loss_ij)
                train_set_sizes.append(train_set_size)
                neighbor_weights.append(neighbor_model.state_dict())

                # update client sampling record
                clients[i].n_sampled[j] += 1
            
            # weighted average of models
            neighbor_weights.append(clients[i].local_model.state_dict())
            train_set_sizes.append(len(clients[i].train_set))
            # do the actual averaging at the end of the round, after calculating similarity scores

            #### calculate new similarity scores
            if parameters['similarity_metric'] == 'inverse_training_loss':
                ij_similarities = [1/(train_losses_ij[i] + inv_epsilon) for i in range(len(train_losses_ij))]
            elif parameters['similarity_metric'] == 'cosine_similarity':
                i_grad_a = clients[i].get_grad_a()
                i_grad_b = clients[i].get_grad_b()
                ij_similarities = []
                for idx, j in enumerate(neighbor_indices_sampled):
                    j_grad_a = clients[j].get_grad_a()
                    j_grad_b = clients[j].get_grad_b()
                    # calculate cosine similarity using torches cosine similarity function
                    cosine_similarity_a = torch.nn.functional.cosine_similarity(torch.tensor(i_grad_a), torch.tensor(j_grad_a), dim=0)
                    cosine_similarity_b = torch.nn.functional.cosine_similarity(torch.tensor(i_grad_b), torch.tensor(j_grad_b), dim=0)
                    ij_similarities.append(parameters['cosine_alpha']*cosine_similarity_a + (1 - parameters['cosine_alpha'])*cosine_similarity_b)

            elif parameters['similarity_metric'] == 'cosine_origin':
                i_grad_origin = clients[i].get_grad_origin()
                ij_similarities = []
                for idx, j in enumerate(neighbor_indices_sampled):
                    j_grad_origin = clients[j].get_grad_origin()
                    # calculate cosine similarity using torches cosine similarity function
                    cosine_similarity = torch.nn.functional.cosine_similarity(torch.tensor(i_grad_origin), torch.tensor(j_grad_origin), dim=0)
                    ij_similarities.append(cosine_similarity)

            elif parameters['similarity_metric'] == 'l2':
                i_grad_origin = clients[i].get_grad_origin()
                ij_similarities = []
                for idx, j in enumerate(neighbor_indices_sampled):
                    j_grad_origin = clients[j].get_grad_origin()
                    l2_distance = np.linalg.norm(i_grad_origin - j_grad_origin)
                    ij_similarities.append(1/(l2_distance + inv_epsilon))

            else:
                # other similarity metrics here
                pass

            #### update similarity scores of client i
            # new direct neighbors
            for idx, j in enumerate(neighbor_indices_sampled):
                clients[i].similarity_scores[j] = ij_similarities[idx]
            # two step neighbors
            # all direct neighbors of client j that are not direct neighbors of client i, and that are not i
            two_step_similarities = [[-1]*2 for _ in clients] #[sij, sjk]
            client_list = np.arange(len(clients))
            for j in neighbor_indices_sampled:
                two_step_neighbors_of_j = list(set(client_list[clients[j].n_sampled > 0]) - set(client_list[clients[i].n_sampled > 0]) - set([i]))
                for k in two_step_neighbors_of_j:
                    # if new
                    if two_step_similarities[k][0] == -1:
                        two_step_similarities[k][0] = clients[i].similarity_scores[j]
                        two_step_similarities[k][1] = clients[j].similarity_scores[k]
                    # if already visisted
                    else:
                        if clients[i].similarity_scores[j] < two_step_similarities[k][0]:
                            two_step_similarities[k][0] = clients[i].similarity_scores[j]
                            two_step_similarities[k][1] = clients[j].similarity_scores[k]
            
            for k in range(len(clients)):
                if two_step_similarities[k][0] != -1:
                    clients[i].similarity_scores[k] = two_step_similarities[k][1]

            # save a copy of the similarity scores to all_similarities
            clients[i].all_similarities.append(copy.deepcopy(clients[i].similarity_scores))

            # save a copy of the neighbors to exchanges_every_round
            clients[i].exchanges_every_round.append(copy.deepcopy(neighbor_indices_sampled))
        
            #### update client priors - can maybe be done in several ways
            clients[i].priors = np.zeros(len(clients)) # set priors to zero

            # if variable tau, update tau
            if parameters['prior_update_rule'] == 'softmax-variable-tau':
                #update tau
                parameters['tau'] = tau_function(round,parameters['tau'],0.2)

            elif parameters['prior_update_rule'] == 'softmax-fixed-entropy':
                # find tau that gives an entropy of of 2 bits for the given similarities
                # entropy = -sum(p*log(p))
                found_tau = find_tau(clients[i].similarity_scores, 2)
                clients[i].all_taus.append(found_tau)
                parameters['tau'] = found_tau

            # do the priors update
            if parameters['prior_update_rule'] == 'softmax-variable-tau' or parameters['prior_update_rule'] == 'softmax' or parameters['prior_update_rule'] == 'softmax-fixed-entropy':
                # Extract non-zero similarity scores
                non_zero_indices = [k for k, score in enumerate(clients[i].similarity_scores) if score > 0]
                non_zero_scores = [clients[i].similarity_scores[j] for j in non_zero_indices]

                # Apply the softmax transformation to non-zero entries
                if non_zero_scores:  # Check if there are any non-zero scores
                    max_score = max(non_zero_scores)
                    exp_scores = np.exp(np.array(non_zero_scores - max_score) * parameters['tau'])
                    
                    # Normalize
                    sum_exp_scores = np.sum(exp_scores)
                    for j, score in zip(non_zero_indices, exp_scores):
                        clients[i].priors[j] = score / sum_exp_scores
                
                clients[i].priors += 1e-6
                clients[i].priors[i] = 0
                clients[i].priors = clients[i].priors / np.sum(clients[i].priors)
            
            # FEDERATED AVERAGING
            new_weights = FedAvg(neighbor_weights,train_set_sizes)
            # save old model
            clients[i].pre_merge_model = copy.deepcopy(clients[i].local_model.state_dict())
            # update client model
            clients[i].local_model.load_state_dict(new_weights)

            if verbose:
                print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))

    return clients

def client_information_exchange_oracle(clients, parameters, verbose=False, round=0, delusion=0):
    '''
    parameters:
    n_sampled: number of neighbors sampled
    prior_update_rule: how to update priors
    similarity_metric: how to measure similarity between clients
    '''
    if verbose:
        print('Starting information exchange round {}'.format(round))

    # check if all clients have stopped early
    all_stopped = True
    for client in clients:
        if not client.early_stopping.is_stopped():
            all_stopped = False
            break
    
    if all_stopped:
        if verbose:
            print('All clients have stopped early, stopping information exchange')
        return clients

    # randomize order of asynchronous communication
    idxs = np.arange(len(clients))
    np.random.shuffle(idxs)
    for i in idxs:
        if verbose:
            if clients[i].early_stopping.is_stopped():
                print('Client {} has stopped early'.format(i))
        if (not clients[i].early_stopping.is_stopped()):
            
            #### sample neighbors

            # if delusion is -1, sample randomly from all clients
            if delusion == -1:
                neighbor_indices_sampled = np.random.choice(list(set(range(len(clients))) - set([i])), 
                                                        size=parameters['nbr_neighbors_sampled'], 
                                                        replace=False)
            else:
                
                # client is delusional if a random number between 0 and 1 is less than delusion
                delusional = np.random.rand() < delusion

                if delusional:
                    neighbor_indices_sampled = np.random.choice(list(set([client.idx for client in clients if client.group != clients[i].group]) - set([i])), 
                                                            size=parameters['nbr_neighbors_sampled'], 
                                                            replace=False)
                else:
                    neighbor_indices_sampled = np.random.choice(list(set([client.idx for client in clients if client.group == clients[i].group]) - set([i])), 
                                                            size=parameters['nbr_neighbors_sampled'], 
                                                            replace=False)

            #### aggregate information from chosen neighbors
            # potentially different model aggregation weightings here
            neighbor_weights = []
            train_set_sizes = []
            for j in neighbor_indices_sampled:
                # validate on neighbor model, get loss and accuracy
                neighbor_model = clients[j].local_model
                train_set_size = len(clients[j].train_set)
                # save stuff
                train_set_sizes.append(train_set_size)
                neighbor_weights.append(neighbor_model.state_dict())

                # update client sampling record
                clients[i].n_sampled[j] += 1
            
            # weighted average of models
            neighbor_weights.append(clients[i].local_model.state_dict())
            train_set_sizes.append(len(clients[i].train_set))
            new_weights = FedAvg(neighbor_weights,train_set_sizes)
            # save old model
            clients[i].pre_merge_model = copy.deepcopy(clients[i].local_model.state_dict())
            # update client model
            clients[i].local_model.load_state_dict(new_weights)

            #### calculate new similarity scores
            
        
            #### update client priors - can maybe be done in several ways
            clients[i].priors = np.zeros(len(clients)) # TODO ska vi nollställa?
            

            if verbose:
                print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))

    return clients

def client_information_exchange_LHO(clients, parameters, verbose=False, round=0):
    '''
    parameters:
    n_sampled: number of neighbors sampled
    similarity_metric: how to measure similarity between clients
    '''
    if verbose:
        print('Starting information exchange round {}'.format(round))

    # check if all clients have stopped early
    all_stopped = True
    for client in clients:
        if not client.early_stopping.is_stopped():
            all_stopped = False
            break
    
    if all_stopped:
        if verbose:
            print('All clients have stopped early, stopping information exchange')
        return clients

    # randomize order of asynchronous communication
    idxs = np.arange(len(clients))
    np.random.shuffle(idxs)
    for i in idxs:
        if verbose:
            if clients[i].early_stopping.is_stopped():
                print('Client {} has stopped early'.format(i))
        if (not clients[i].early_stopping.is_stopped()):

            if round < 3:
                pass
            elif round == 3:
                # look
                similarities = clients[i].measure_all_similarities(clients, parameters['similarity_metric'], alpha=0, store=False)
                clients[i].similarity_scores = similarities
            elif round > 3:
                # find top k neighbors
                candidates = np.argsort(clients[i].similarity_scores)[-9:]
                # sample randomly from candidates
                neighbor_indices_sampled = np.random.choice(candidates, 
                                                            size=parameters['nbr_neighbors_sampled'], 
                                                            replace=False)

                #### aggregate information from chosen neighbors
                # potentially different model aggregation weightings here
                neighbor_weights = []
                train_set_sizes = []
                for j in neighbor_indices_sampled:
                    # validate on neighbor model, get loss and accuracy
                    neighbor_model = clients[j].local_model
                    train_set_size = len(clients[j].train_set)
                    # save stuff
                    train_set_sizes.append(train_set_size)
                    neighbor_weights.append(neighbor_model.state_dict())

                    # update client sampling record
                    clients[i].n_sampled[j] += 1
                
                # weighted average of models
                neighbor_weights.append(clients[i].local_model.state_dict())
                train_set_sizes.append(len(clients[i].train_set))
                new_weights = FedAvg(neighbor_weights,train_set_sizes)
                clients[i].local_model.load_state_dict(new_weights)

                #### calculate new similarity scores
                
            
                #### update client priors - can maybe be done in several ways
                clients[i].priors = np.zeros(len(clients)) # TODO ska vi nollställa?
            

                if verbose:
                    print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))

    return clients

def get_delusional_clients(clients, nbr_deluded_clients, deluded_group=0):
    nbr_clients_delusion = nbr_deluded_clients
    group_with_delusion = deluded_group
    delusional_client_idxs = np.random.choice([client.idx for client in clients if client.group == group_with_delusion], nbr_clients_delusion, replace=False)
    return delusional_client_idxs

def client_information_exchange_some_delusion(clients, parameters, verbose=False, round=0):
    '''
    parameters:
    n_sampled: number of neighbors sampled
    prior_update_rule: how to update priors
    similarity_metric: how to measure similarity between clients
    '''
    # 'nbr_neighbors_sampled': args.nbr_neighbors_sampled,
    # 'start_delusion': 20,
    # 'delusional_client_idxs': delusional_client_idxs
    
    if verbose:
        print('Starting information exchange round {}'.format(round))

    # check if all clients have stopped early
    all_stopped = True
    for client in clients:
        if not client.early_stopping.is_stopped():
            all_stopped = False
            break
    
    if all_stopped:
        if verbose:
            print('All clients have stopped early, stopping information exchange')
        return clients
    
    # randomize order of asynchronous communication
    idxs = np.arange(len(clients))
    np.random.shuffle(idxs)
    for i in idxs:
        if verbose:
            if clients[i].early_stopping.is_stopped():
                print('Client {} has stopped early'.format(i))
        if (not clients[i].early_stopping.is_stopped()):
            
            #### sample neighbors
            # client is delusional if a random number between 0 and 1 is less than delusion

            if round < parameters['start_delusion']:
                neighbor_indices_sampled = np.random.choice(list(set([client.idx for client in clients if client.group == clients[i].group]) - set([i])), 
                                                        size=parameters['nbr_neighbors_sampled'], 
                                                        replace=False)
            else:
                if i in parameters['delusional_client_idxs']:
                    neighbor_indices_sampled = np.random.choice(list(set([client.idx for client in clients if client.group != clients[i].group]) - set([i])), 
                                                        size=parameters['nbr_neighbors_sampled'], 
                                                        replace=False)
                else:
                    neighbor_indices_sampled = np.random.choice(list(set([client.idx for client in clients if client.group == clients[i].group]) - set([i])), 
                                                        size=parameters['nbr_neighbors_sampled'], 
                                                        replace=False)

            #### aggregate information from chosen neighbors
            # potentially different model aggregation weightings here
            neighbor_weights = []
            train_set_sizes = []
            for j in neighbor_indices_sampled:
                # validate on neighbor model, get loss and accuracy
                neighbor_model = clients[j].local_model
                train_set_size = len(clients[j].train_set)
                # save stuff
                train_set_sizes.append(train_set_size)
                neighbor_weights.append(neighbor_model.state_dict())

                # update client sampling record
                clients[i].n_sampled[j] += 1
            
            # weighted average of models
            neighbor_weights.append(clients[i].local_model.state_dict())
            train_set_sizes.append(len(clients[i].train_set))
            new_weights = FedAvg(neighbor_weights,train_set_sizes)
            clients[i].local_model.load_state_dict(new_weights)

            #### calculate new similarity scores
            
        
            #### update client priors - can maybe be done in several ways
            clients[i].priors = np.zeros(len(clients)) # TODO ska vi nollställa?
            

            if verbose:
                print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))
                if i in parameters['delusional_client_idxs'] and round >= parameters['start_delusion']:
                    print('Client {} has become delusional'.format(i))

    return clients


### BIG NOTE, PANM does not have new fancy similarity scores, and neither does it have torches cosine similarity function

def NSMC(clients, N, i, nbr_neighbors_sampled):
    # N is a list of tuples (client, similarity)
    # neighbors is first index of each tuple
    neighbors = [tup[0] for tup in N]
    valid_samples = list(set(range(len(clients))) - set(neighbors) - set([i]))
    # return random uniform sample from valid samples
    return np.random.choice(valid_samples, size=nbr_neighbors_sampled, replace=False)

def NAEM(clients, B, i, parameters, verbose=False):

    inv_epsilon = 1e-6

    # B is a list of tuples (client, similarity)
    unsampled_neighbors_idxs = NSMC(clients, B, i, parameters['nbr_neighbors_sampled']) # C in paper

    # gamma_ij initialization for each new random client
    gamma_ij = []
    for j in unsampled_neighbors_idxs:
        gamma_ij.append([j,1]) # [client index, gamma_ij]

    # for cosine
    i_grad_a = clients[i].get_grad_a()
    i_grad_b = clients[i].get_grad_b()

    # calculate similarity scores
    unsampled_neighbors = []
    for j in unsampled_neighbors_idxs:
        if parameters['similarity_metric'] == 'inverse_training_loss':
            train_loss_ij, train_acc_ij = clients[i].validate(clients[j].local_model, train_set=True)
            unsampled_neighbors.append((j,1/train_loss_ij + inv_epsilon))
        elif parameters['similarity_metric'] == 'cosine_similarity':
            j_grad_a = clients[j].get_grad_a()
            j_grad_b = clients[j].get_grad_b()
            similarity = parameters['cosine_alpha']*np.dot(i_grad_a,j_grad_a)/(np.linalg.norm(i_grad_a)*np.linalg.norm(j_grad_a))
            + (1 - parameters['cosine_alpha'])*np.dot(i_grad_b,j_grad_b)/(np.linalg.norm(i_grad_b)*np.linalg.norm(j_grad_b))
            clients[i].N.append((j,similarity))
        
    # sample from bag
    if len(B) > parameters['nbr_neighbors_sampled']:
        bag_idxs = np.random.choice(len(B), size=parameters['nbr_neighbors_sampled'], replace=False) # S in paper
        bag_samples = [B[i] for i in bag_idxs]
        left_in_bag = [item for idx, item in enumerate(B) if idx not in bag_idxs]
        # b-s ska returnas också
    else:
        bag_samples = B
        left_in_bag = []

    
    # M is the union of C and S
    M = list(set(unsampled_neighbors) | set(bag_samples))
    
    # gamma_ij initialization for each client in bag
    for sample in bag_samples: 
        j = sample[0]
        gamma_ij.append([j,0]) # [client index, gamma_ij]

    changing = True
    not_to_many_rounds = 0
    while(changing):       
        gamma_ij_old = deepcopy(gamma_ij)
        #### E-step ####
        # estimate mu_0, sigma_0 beta_0, mu_1, sigma_1, beta_1
        # mu_r = sum(gamma_jr*similarity_j) / sum(gamma_jr)
        # beta_r = sum(gamma_jr)/len(M)
        # sigma_r² = sum(gamma_jr*(similarity_j - mu_r)²) / sum(gamma_jr)
        cluster_0_client_indexes = [gamma[0] for gamma in gamma_ij if gamma[1] == 0]
        simalarity_cluster_0 = [m[1] for m in M if m[0] in cluster_0_client_indexes] 

        if simalarity_cluster_0:
            mu_0 = np.sum(simalarity_cluster_0)/len(simalarity_cluster_0)
            beta_0 = len(simalarity_cluster_0)/len(M)        
            var_0 = (sum((element - mu_0) ** 2 for element in simalarity_cluster_0))/len(simalarity_cluster_0) # sigma_0²
        else: # if simalrity cluser 0 is empty
            gamma_ij = []
            for sample in bag_samples: 
                j = sample[0]
                gamma_ij.append([j,0])
                if verbose:
                    print('cluster 0 empty, breaking loop')
            break

        cluster_1_client_indexes = [gamma[0] for gamma in gamma_ij if gamma[1] == 1]
        simalarity_cluster_1 = [m[1] for m in M if m[0] in cluster_1_client_indexes]

        if simalarity_cluster_1:
            mu_1 = np.sum(simalarity_cluster_1)/len(simalarity_cluster_1)
            beta_1 = len(simalarity_cluster_1)/len(M)
            var_1 = (sum((element - mu_1) ** 2 for element in simalarity_cluster_1))/len(simalarity_cluster_1) # sigma_1²
        else: #if similarity cluster 1 is empty
            break

        #### M-step ####
        # update gamma_ij 
        for m_index, m_sim in M:
            row_index_list = [idx for idx, item in enumerate(gamma_ij) if item[0] == m_index]
            row_index = row_index_list[0]
            
            if beta_0*norm.pdf(m_sim, mu_0, np.sqrt(var_0)) > beta_1*norm.pdf(m_sim, mu_1, np.sqrt(var_1)):
                gamma_ij[row_index][1] = 0
            else:
                gamma_ij[row_index][1] = 1
        
        if sorted(gamma_ij) == sorted(gamma_ij_old):
            changing = False

        if not_to_many_rounds >= 100:
            changing = False
            if verbose:
                print('NAEM did not converge in 100 rounds, breaking loop. (client {})'.format(i))
        not_to_many_rounds += 1
    if verbose:
        print('client {} finished EM in {} rounds'.format(i, not_to_many_rounds))
 
    worthy_clients_index = [i[0] for i in gamma_ij if i[1] == 0]
    return [tuple(row) for row in M if row[0] in worthy_clients_index] + left_in_bag

def client_information_exchange_PANM(clients, parameters, verbose=False, round=0):
    '''
    parameters:
    n_sampled: number of neighbors sampled
    prior_update_rule: how to update priors
    similarity_metric: how to measure similarity between clients
    T1: how many rounds to use NSMC
    T2: how many rounds to use NAEM
    '''

    inv_epsilon = 1e-6

    # check if all clients have stopped early
    all_stopped = True
    for client in clients:
        if not client.early_stopping.is_stopped():
            all_stopped = False
            break
    
    if all_stopped:
        if verbose:
            print('All clients have stopped early, stopping information exchange')
        return clients

    if verbose:
        print('Starting information exchange round {}'.format(round))

    # randomize order of asynchronous communication
    idxs = np.arange(len(clients))
    np.random.shuffle(idxs)
    for i in idxs:
        if verbose:
            if clients[i].early_stopping.is_stopped():
                print('Client {} has stopped early'.format(i))
        if (not clients[i].early_stopping.is_stopped()):
            
            # check T1 or T2
            if round < parameters['T1']:
                # sample neighbors uniformly, but dont sample neighbors already in N or oneself
                neighbor_indices_sampled = NSMC(clients, clients[i].N, i, parameters['nbr_neighbors_sampled'])
                
                # for cosine
                i_grad_a = clients[i].get_grad_a()
                i_grad_b = clients[i].get_grad_b()

                # calculate similarity scores
                # for each neighbor
                for j in neighbor_indices_sampled:
                    if parameters['similarity_metric'] == 'inverse_training_loss':
                        train_loss_ij, train_acc_ij = clients[i].validate(clients[j].local_model, train_set=True)
                        clients[i].N.append((j,1/(train_loss_ij + inv_epsilon)))
                        clients[i].similarity_scores[j] = 1/(train_loss_ij + inv_epsilon)
                    elif parameters['similarity_metric'] == 'cosine_similarity':
                        j_grad_a = clients[j].get_grad_a()
                        j_grad_b = clients[j].get_grad_b()
                        cosine_similarity_a = torch.nn.functional.cosine_similarity(torch.tensor(i_grad_a), torch.tensor(j_grad_a), dim=0)
                        cosine_similarity_b = torch.nn.functional.cosine_similarity(torch.tensor(i_grad_b), torch.tensor(j_grad_b), dim=0)
                        similarity = parameters['cosine_alpha']*cosine_similarity_a + (1 - parameters['cosine_alpha'])*cosine_similarity_b
                        clients[i].N.append((j,similarity))
                        clients[i].similarity_scores[j] = similarity

                # find top-k neighbors
                clients[i].N.sort(key=lambda x: x[1], reverse=True)
                clients[i].N = clients[i].N[:parameters['nbr_neighbors_sampled']]

                # aggregate information from chosen neighbors
                neighbor_weights = []
                train_set_sizes = []
                neighbor_indices_sampled = []
                for tup in clients[i].N:
                    j = tup[0]
                    neighbor_indices_sampled.append(j)
                    neighbor_model = clients[j].local_model
                    train_set_size = len(clients[j].train_set)
                    train_set_sizes.append(train_set_size)
                    neighbor_weights.append(neighbor_model.state_dict())
                    clients[i].n_sampled[j] += 1

                # save a copy of the similarity scores to all_similarities
                clients[i].all_similarities.append(copy.deepcopy(clients[i].similarity_scores))

                # save a copy of the neighbors to exchanges_every_round
                clients[i].exchanges_every_round.append(copy.deepcopy(neighbor_indices_sampled))

                # weighted average of models
                neighbor_weights.append(clients[i].local_model.state_dict())
                train_set_sizes.append(len(clients[i].train_set))
                
                new_weights = FedAvg(neighbor_weights,train_set_sizes)
                clients[i].local_model.load_state_dict(new_weights)

                
            else: # T2
                # if first round of T2, initialize bag
                if round == parameters['T1']:
                    clients[i].B = clients[i].N
                
                if round%parameters['NAEM_frequency'] == 0: # perform NAEM
                    clients[i].B = NAEM(clients, clients[i].B, i, parameters, verbose=verbose)
                    if verbose:
                        print('client {} - bag: {}'.format(i, clients[i].B))
                else: # dont perform NAEM.
                    pass

                # N uniform sample from B
                if len(clients[i].B) > parameters['nbr_neighbors_sampled']:
                    N_idxs = np.random.choice(len(clients[i].B), 
                                    size=parameters['nbr_neighbors_sampled'], 
                                    replace=False)
                
                    N = [clients[i].B[n] for n in N_idxs]
                else:
                    N = clients[i].B
                # aggregate information from chosen neighbors
                neighbor_weights = []
                train_set_sizes = []
                neighbor_indices_sampled = []
                for tup in N:
                    j = tup[0]
                    neighbor_indices_sampled.append(j)
                    neighbor_model = clients[j].local_model
                    train_set_size = len(clients[j].train_set)
                    train_set_sizes.append(train_set_size)
                    neighbor_weights.append(neighbor_model.state_dict())
                    clients[i].n_sampled[j] += 1
                
                # weighted average of models
                neighbor_weights.append(clients[i].local_model.state_dict())
                train_set_sizes.append(len(clients[i].train_set))

                new_weights = FedAvg(neighbor_weights,train_set_sizes)
                clients[i].local_model.load_state_dict(new_weights)
            if verbose:
                print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))
    
    return clients

