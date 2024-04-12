import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import random

def generate_regression_multi(theta, n, sigma):
    d = len(theta)
    X = np.random.uniform(-10, 10, (n, d))
    Y = np.dot(X, theta) + np.random.normal(0, sigma, n)
    return X, Y

def FedAvg(w,alpha):
    alpha = alpha/np.sum(alpha) #normalize alpha
    w_avg = copy.deepcopy(w[0])
    n_clients = len(w)
    
    for l in w_avg.keys():
        w_avg[l] = w_avg[l] - w_avg[l]

    for l, layer in enumerate(w_avg.keys()): #for each layer
        w_kl = []
        for k in range(0,n_clients): #for each client
            w_avg[layer] += alpha[k]*w[k][layer]
    return w_avg

class FederatedClient():
    def __init__(self, model, criterion, train_loader, test_loader=None):
        self.model = model
        #self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    def train(self, num_epochs, learning_rate, fed_alg, mu=0):
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.model.train()
        global_model = copy.deepcopy(self.model)
        train_loss = []
        train_acc = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_acc = 0
            loss1_sum = 0
            loss2_sum = 0
            for i, (x, y) in enumerate(self.train_loader):
                #x = x.to('cuda:0')
                #y = y.to('cuda:0')
                optimizer.zero_grad()
                outputs = self.model(x)
                if(fed_alg=='fedavg'):
                    loss = self.criterion(outputs, y)
                elif(fed_alg=='fedprox'):
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += torch.norm(param - global_param, p=2)
                    loss1 = self.criterion(outputs, y) 
                    loss2 = mu/2 * proximal_term
                    loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += (outputs.argmax(1) == y).sum().item()
            train_loss.append(epoch_loss / len(self.train_loader))
            #train_acc.append(epoch_acc / len(self.train_loader))
        
        return self.model, train_loss
    
    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                test_loss += loss.item()
        
        return test_loss / len(self.test_loader)
    
    def get_model(self):
        return self.model
    

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  
        
    def forward(self, x):
        out = self.linear(x)
        return out
    

def train(model, optimizer, criterion, n_communication, train_loader, val_loader=None):
    model.train()
    train_loss = []
    val_loss = []
    for epoch in range(n_communication):
        epoch_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss.append(epoch_loss / len(train_loader))
        
        if(val_loader):
            model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                for x, y in val_loader:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    epoch_val_loss += loss.item()
                val_loss.append(epoch_val_loss / len(val_loader))
            model.train()
        
    return model, train_loss, val_loss

def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
        
    return test_loss / len(test_loader)


def run_experiment(n_exp, n_clients, input_dim, output_dim, mu_U1, mu_U2, mu_U3, sigma_u1, sigma_u2, sigma_u3, n_client_data, n_test, num_local_epochs, n_communication, lr):

    # set seed
    np.random.seed(n_exp)
    torch.manual_seed(n_exp)
    # set python seed
    random.seed(n_exp)

    print("Experiment: ", n_exp+1)
    # initialize models
    models = [LinearRegression(input_dim, output_dim) for _ in range(n_clients)]

    clients_oracle = []
    clients_random = []
    clients_DAC = []

    # creating clients
    for i in range(n_clients):
        if(i < 1/3*n_clients):
            mu_U = mu_U1
            sigma_U = sigma_u1
        if(i >= 1/3*n_clients and i < 2/3*n_clients):
            mu_U = mu_U2
            sigma_U = sigma_u2
        if(i >= 2/3*n_clients):
            mu_U = mu_U3
            sigma_U = sigma_u3

        X, Y = generate_regression_multi(mu_U, n_client_data, sigma_U)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float().reshape(-1,1)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

        clients_random.append(FederatedClient(copy.deepcopy(models[i]), nn.MSELoss(), loader))
        clients_oracle.append(FederatedClient(copy.deepcopy(models[i]), nn.MSELoss(), loader))
        clients_DAC.append(FederatedClient(copy.deepcopy(models[i]), nn.MSELoss(), loader))


    # creating testloaders
    X1_test, Y1_test = generate_regression_multi(mu_U1, n_test, sigma_u1)
    X2_test, Y2_test = generate_regression_multi(mu_U2, n_test, sigma_u2)
    X3_test, Y3_test = generate_regression_multi(mu_U3, n_test, sigma_u3)

    X1_test = torch.from_numpy(X1_test).float()
    Y1_test = torch.from_numpy(Y1_test).float().reshape(-1,1)
    test1_dataset = torch.utils.data.TensorDataset(X1_test, Y1_test)
    test1_loader = torch.utils.data.DataLoader(dataset=test1_dataset, batch_size=128, shuffle=False)

    X2_test = torch.from_numpy(X2_test).float()
    Y2_test = torch.from_numpy(Y2_test).float().reshape(-1,1)
    test2_dataset = torch.utils.data.TensorDataset(X2_test, Y2_test)
    test2_loader = torch.utils.data.DataLoader(dataset=test2_dataset, batch_size=128, shuffle=False)

    X3_test = torch.from_numpy(X3_test).float()
    Y3_test = torch.from_numpy(Y3_test).float().reshape(-1,1)
    test3_dataset = torch.utils.data.TensorDataset(X3_test, Y3_test)
    test3_loader = torch.utils.data.DataLoader(dataset=test3_dataset, batch_size=128, shuffle=False)

    #initialize models

    mean_loss_random = []
    mean_loss_oracle = []

    alpha = np.ones(n_clients)
    alpha = alpha/np.sum(alpha)

    for k in range(n_communication):
        models_fedavg_random, models_fedavg_oracle = [], []
        for i in range(n_clients):
            model_random, train_loss_random = clients_random[i].train(num_local_epochs,lr,'fedavg')
            model_oracle, train_loss_oracle = clients_oracle[i].train(num_local_epochs,lr,'fedavg')

            models_fedavg_random.append(model_random.state_dict())
            models_fedavg_oracle.append(model_oracle.state_dict())


        w_global_model_fedavg = FedAvg(models_fedavg_random, alpha)
        #one model per domain
        w_global_U1 = FedAvg(models_fedavg_oracle[:int((1/3)*n_clients)], alpha[:int((1/3)*n_clients)])
        w_global_U2 = FedAvg(models_fedavg_oracle[int((1/3)*n_clients):int((2/3)*n_clients)], alpha[int((1/3)*n_clients):int((2/3)*n_clients)])
        w_global_U3 = FedAvg(models_fedavg_oracle[int((2/3)*n_clients):], alpha[int((2/3)*n_clients):])

        for i in range(n_clients):
            clients_random[i].model.load_state_dict(copy.deepcopy(w_global_model_fedavg))
            if(i < (1/3)*n_clients):
                clients_oracle[i].model.load_state_dict(copy.deepcopy(w_global_U1))
            if(i >= (1/3)*n_clients and i < (2/3)*n_clients):
                clients_oracle[i].model.load_state_dict(copy.deepcopy(w_global_U2))
            if(i >= (2/3)*n_clients):
                clients_oracle[i].model.load_state_dict(copy.deepcopy(w_global_U3))

    random_fed.load_state_dict(copy.deepcopy(w_global_model_fedavg))
    oracle_fed1.load_state_dict(copy.deepcopy(w_global_U1))
    oracle_fed2.load_state_dict(copy.deepcopy(w_global_U2))
    oracle_fed3.load_state_dict(copy.deepcopy(w_global_U3))


    #train one model per domain
    optimizer1 = optim.SGD(model1.parameters(), lr)
    optimizer2 = optim.SGD(model2.parameters(), lr)
    optimizer3 = optim.SGD(model3.parameters(), lr)

    model1, train_loss1, val_loss1 = train(model1, optimizer1, nn.MSELoss(), n_communication, clients_random[0].train_loader, val_loader=None)
    model2, train_loss2, val_loss2 = train(model2, optimizer2, nn.MSELoss(), n_communication, clients_random[int((1/3)*n_clients)].train_loader, val_loader=None)
    model3, train_loss3, val_loss3 = train(model3, optimizer3, nn.MSELoss(), n_communication, clients_random[int((1/3)*n_clients)].train_loader, val_loader=None)

    test_loaders = [test1_loader, test2_loader, test3_loader]

    random_MSE_list = []
    oracle_MSE_list = []
    U1_MSE_list = []
    U2_MSE_list = []
    U3_MSE_list = []
    for i in range(len(test_loaders)):
        random_MSE = test(clients_random[0].model, nn.MSELoss(), test_loaders[i])
        if(i==0):
            oracle_MSE = test(clients_oracle[0].model, nn.MSELoss(), test_loaders[i])
        if(i==1):
            oracle_MSE = test(clients_oracle[int((1/3)*n_clients)].model, nn.MSELoss(), test_loaders[i])
        if(i==2):
            oracle_MSE = test(clients_oracle[int((2/3)*n_clients)].model, nn.MSELoss(), test_loaders[i])
        
        U1_MSE = test(model1, nn.MSELoss(), test_loaders[i])
        U2_MSE = test(model2, nn.MSELoss(), test_loaders[i])
        U3_MSE = test(model3, nn.MSELoss(), test_loaders[i])
        random_MSE_list.append(random_MSE)
        oracle_MSE_list.append(oracle_MSE)
        U1_MSE_list.append(U1_MSE)
        U2_MSE_list.append(U2_MSE)
        U3_MSE_list.append(U3_MSE)

    in_domain_test_scores = [U1_MSE_list[0], U2_MSE_list[1], U3_MSE_list[2]]

    U1_MSE_test_list.append(U1_MSE_list[0])
    U2_MSE_test_list.append(U2_MSE_list[1])
    U3_MSE_test_list.append(U3_MSE_list[2])

    U1_random_list.append(random_MSE_list[0])
    U2_random_list.append(random_MSE_list[1])
    U3_random_list.append(random_MSE_list[2])

    U1_oracle_list.append(oracle_MSE_list[0])
    U2_oracle_list.append(oracle_MSE_list[1])
    U3_oracle_list.append(oracle_MSE_list[2])