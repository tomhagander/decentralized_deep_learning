from utils.toy_regression_utils import *

U1_MSE_test_list, U2_MSE_test_list, U3_MSE_test_list = [], [], []
U1_random_list, U2_random_list, U3_random_list = [], [], []
U1_oracle_list, U2_oracle_list, U3_oracle_list = [], [], []

mu_U1 = np.random.uniform(-10, 10, 10)
mu_U2 = np.random.uniform(-10, 10, 10)
mu_U3 = np.random.uniform(-10, 10, 10)

sigma_u1 = 3
sigma_u2 = 3
sigma_u3 = 3

n_client_data = 100
n_test = 1000

num_local_epochs = 1 #local epochs
n_communication = 10
n_clients = 60
lr = 0.003
input_dim = 10
output_dim = 1

for n_exp in range(10):
    run_experiment(n_exp)

for n_exp in range(10):
    print("Experiment: ", n_exp+1)
    #initalize a model
    model = LinearRegression(input_dim, output_dim)
    model1 = LinearRegression(input_dim, output_dim)
    model2 = LinearRegression(input_dim, output_dim)
    model3 = LinearRegression(input_dim, output_dim)

    random_fed = LinearRegression(input_dim, output_dim)
    oracle_fed1 = LinearRegression(input_dim, output_dim)
    oracle_fed2 = LinearRegression(input_dim, output_dim)
    oracle_fed3 = LinearRegression(input_dim, output_dim)

    clients_random, clients_oracle = [], []
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

        clients_random.append(FederatedClient(copy.deepcopy(model), nn.MSELoss(), loader))
        clients_oracle.append(FederatedClient(copy.deepcopy(model), nn.MSELoss(), loader))

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

    mean_loss_random = []
    mean_loss_oracle = []

    alpha = np.ones(n_clients)
    alpha = alpha/np.sum(alpha)

    for k in range(n_communication):
        train_losses_random = []
        train_losses_oracle = []
        models_fedavg_random, models_fedavg_oracle = [], []
        for i in range(n_clients):
            model_random, train_loss_random = clients_random[i].train(num_local_epochs,lr,'fedavg')
            model_oracle, train_loss_oracle = clients_oracle[i].train(num_local_epochs,lr,'fedavg')

            models_fedavg_random.append(model_random.state_dict())
            models_fedavg_oracle.append(model_oracle.state_dict())

            train_losses_random.append(train_loss_random[-1])
            train_losses_oracle.append(train_loss_oracle[-1])

        
        mean_loss_random.append(np.mean(train_losses_random))
        mean_loss_oracle.append(np.mean(train_losses_oracle))
        #print("Round: ", k+1, "\tAverage train loss: ", mean_loss[-1])

        w_global_model_fedavg = FedAvg(models_fedavg_random, alpha)
        #one model per domain
        w_global_U1 = FedAvg(models_fedavg_oracle[:3], alpha[:3])
        w_global_U2 = FedAvg(models_fedavg_oracle[3:6], alpha[3:6])
        w_global_U3 = FedAvg(models_fedavg_oracle[6:], alpha[6:])

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

print("MSE Domain 1: ", np.mean(U1_MSE_test_list))
print("MSE Random D1: ", np.mean(U1_random_list))
print("MSE Oracle D1: ", np.mean(U1_oracle_list))
print("-"*50)
print("MSE Domain 2: ", np.mean(U2_MSE_test_list))
print("MSE Random D2: ", np.mean(U2_random_list))
print("MSE Oracle D2: ", np.mean(U2_oracle_list))
print("-"*50)
print("MSE Domain 3: ", np.mean(U3_MSE_test_list))
print("MSE Random D3: ", np.mean(U3_random_list))
print("MSE Oracle D3: ", np.mean(U3_oracle_list))