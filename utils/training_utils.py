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
        if not client.early_stopping.is_stopped():
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
                neighbor_indices_sampled = np.random.choice(list(set(range(len(clients))) - set([i]), 
                                                        size=parameters['nbr_neighbors_sampled'], 
                                                        replace=False))
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
            clients[i].local_model.load_state_dict(new_weights)

            #### calculate new similarity scores
            
        
            #### update client priors - can maybe be done in several ways
            clients[i].priors = np.zeros(len(clients)) # TODO ska vi nollställa?
            

            if verbose:
                print('Client {} informaton exchange round {} done. Exchanged with {}'.format(i, round, neighbor_indices_sampled))

    return clients

def NSMC(clients, N, i, nbr_neighbors_sampled):
    # N is a list of tuples (client, similarity)
    # neighbors is first index of each tuple
    neighbors = [tup[0] for tup in N]
    valid_samples = list(set(clients) - set(neighbors) - set([i]))
    # return random uniform sample from valid samples
    return np.random.choice(valid_samples, size=nbr_neighbors_sampled, replace=False)

def NAEM(clients, B, i, parameters):
    # B is a list of tuples (client, similarity)
    unsampled_neighbors_idxs = NSMC(clients, B, i, parameters['nbr_neighbors_sampled']) # C in paper
    # calculate similarity scores
    unsampled_neighbors = []
    for j in unsampled_neighbors_idxs:
        if parameters['similarity_metric'] == 'inverse_training_loss':
            train_loss_ij, train_acc_ij = clients[i].validate(clients[j].local_model, train_set=True)
            unsampled_neighbors.append((j,1/train_loss_ij))
        elif parameters['similarity_metric'] == 'cosine_similarity':
            pass #TODO

    # sample from bag
    bag_samples = np.random.choice(B, size=parameters['nbr_neighbors_sampled'], replace=False) # S in paper
    # M is the union of C and S
    M = list(set(unsampled_neighbors_idxs) | set(bag_samples))
    # TODO
    return B

def client_information_exchange_PANM(clients, parameters, verbose=False, round=0):
    '''
    parameters:
    n_sampled: number of neighbors sampled
    prior_update_rule: how to update priors
    similarity_metric: how to measure similarity between clients
    T1: how many rounds to use NSMC
    T2: how many rounds to use NAEM
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
            
            # check T1 or T2
            if round < parameters['T1']:
                # sample neighbors uniformly, but dont sample neighbors already in N or oneself
                neighbor_indices_sampled = NSMC(clients, clients[i].N, i, parameters['nbr_neighbors_sampled'])
                
                # calculate similarity scores
                # for each neighbor
                for j in neighbor_indices_sampled:
                    if parameters['similarity_metric'] == 'inverse_training_loss':
                        train_loss_ij, train_acc_ij = clients[i].validate(clients[j].local_model, train_set=True)
                        clients[i].N.append((j,1/train_loss_ij))
                    elif parameters['similarity_metric'] == 'cosine_similarity':
                        pass #TODO

                # find top-k neighbors
                clients[i].N.sort(key=lambda x: x[1])
                clients[i].N = clients[i].N[:parameters['nbr_neighbors_sampled']]

                # aggregate information from chosen neighbors
                neighbor_weights = []
                train_set_sizes = []
                for tup in clients[i].N:
                    j = tup[0]
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

                
            else: # T2
                # if first round of T2, initialize bag
                if round == parameters['T1']:
                    clients[i].B = clients[i].N
                
                if round%parameters['NAEM_frequency'] == 0: # perform NAEM
                    clients[i].B = NAEM(clients[i].B)
                else: # dont perform NAEM
                    pass

                # N uniform sample from B
                N = np.random.choice(clients[i].B, 
                                    size=parameters['nbr_neighbors_sampled'], 
                                    replace=False)
                # aggregate information from chosen neighbors
                neighbor_weights = []
                train_set_sizes = []
                for tup in N:
                    j = tup[0]
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

