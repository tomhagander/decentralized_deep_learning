import numpy as np
import copy

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

# trains all clients locally, one by one, for the set number of local epochs
def train_clients_locally(clients, nbr_local_epochs, verbose=False):
    if verbose:
        print('Starting local training')
    for client in clients:
        client.train(nbr_local_epochs)
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

    if verbose:
        print('Starting information exchange round {}'.format(round))

    # randomize order of asynchronous communication
    idxs = np.arange(len(clients))
    np.random.shuffle(idxs)
    for i in idxs:
        if verbose:
            if clients[i].stopped_early:
                print('Client {} has stopped early'.format(i))
        if (not clients[i].stopped_early):
            
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
            new_weights = FedAvg(neighbor_weights,train_set_sizes)
            clients[i].local_model.load_state_dict(new_weights)

            #### calculate new similarity scores
            if parameters['similarity_metric'] == 'inverse_training_loss':
                ij_similarities = [1/(train_losses_ij[i]) for i in range(len(train_losses_ij))]
            elif parameters['similarity_metric'] == 'cosine_similarity':
                i_grad_a = clients[i].get_grad_a()
                i_grad_b = clients[i].get_grad_b()
                ij_similarities = []
                for idx, j in enumerate(neighbor_indices_sampled):
                    j_grad_a = clients[j].get_grad_a()
                    j_grad_b = clients[j].get_grad_b()
                    ij_similarities.append(parameters['cosine_alpha']*np.dot(i_grad_a,j_grad_a)/(np.linalg.norm(i_grad_a)*np.linalg.norm(j_grad_a))
                    + (1 - parameters['cosine_alpha'])*np.dot(i_grad_b,j_grad_b)/(np.linalg.norm(i_grad_b)*np.linalg.norm(j_grad_b)) + 1)
                    # shifted to be in [0,2] here, should we do this?? Only done to make softmax not be wierd
                    # print info
                    # print('Client {} type a similarity to {}: {}'.format(i, j, np.dot(i_grad_a,j_grad_a)/(np.linalg.norm(i_grad_a)*np.linalg.norm(j_grad_a))))
                    # print('Client {} type b similarity to {}: {}'.format(i, j, np.dot(i_grad_b,j_grad_b)/(np.linalg.norm(i_grad_b)*np.linalg.norm(j_grad_b))))


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

            # save all similarity scores
            clients[i].all_similarities.append(clients[i].similarity_scores)
        
            #### update client priors - can maybe be done in several ways
            clients[i].priors = np.zeros(len(clients)) # TODO ska vi nollställa?
            if parameters['prior_update_rule'] == 'softmax-variable-tau':
                #update tau
                parameters['tau'] = tau_function(round,parameters['tau'],0.2) # this line differs from softmax
                for j in range(len(clients)):
                    if clients[i].similarity_scores[j] > 0:
                        clients[i].priors[j] = np.exp(clients[i].similarity_scores[j]*parameters['tau'])
                # normalize
                clients[i].priors = clients[i].priors/np.sum(clients[i].priors)
            elif parameters['prior_update_rule'] == 'softmax':
                for j in range(len(clients)):
                    if clients[i].similarity_scores[j] > 0:
                        clients[i].priors[j] = np.exp(clients[i].similarity_scores[j]*parameters['tau'])
                # normalize
                clients[i].priors = clients[i].priors/np.sum(clients[i].priors)
            else:
                # other prior update rules here
                pass

            if verbose:
                print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))

    return clients

def client_information_exchange_oracle(clients, parameters, verbose=False, round=0, delusional=False):
    '''
    parameters:
    n_sampled: number of neighbors sampled
    prior_update_rule: how to update priors
    similarity_metric: how to measure similarity between clients
    '''

    if verbose:
        print('Starting information exchange round {}'.format(round))

    # randomize order of asynchronous communication
    idxs = np.arange(len(clients))
    np.random.shuffle(idxs)
    for i in idxs:
        if verbose:
            if clients[i].stopped_early:
                print('Client {} has stopped early'.format(i))
        if (not clients[i].stopped_early):
            
            #### sample neighbors
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
            clients[i].local_model.load_state_dict(new_weights)

            #### calculate new similarity scores
            
        
            #### update client priors - can maybe be done in several ways
            clients[i].priors = np.zeros(len(clients)) # TODO ska vi nollställa?
            

            if verbose:
                print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))

    return clients