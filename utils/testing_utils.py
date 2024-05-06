import torch
import numpy as np
from utils.toy_regression_utils import generate_regression_multi
import time

def test_model(model, testloader):
    # Set the model to evaluation

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for accuracy calculation
    total_correct = 0
    total_samples = 0

    # Iterate over the testloader_vehicle
    for images, labels in testloader:
        # Move images and labels to the device
        images = images.to(device)
        labels = labels.to(device)
    
        # Forward pass
        with torch.no_grad():
            outputs = model(images)
        
        # Get the predicted labels
        _, predicted = torch.max(outputs, 1)
        
        # Update the total number of samples
        total_samples += labels.size(0)
        
        # Update the total number of correct predictions
        total_correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * total_correct / total_samples

    return accuracy

def test_toy_model(model, testloader):
    # Set the model to evaluation

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for accuracy calculation
    total_loss = 0
    total_samples = 0

    # Iterate over the testloader_vehicle
    for data in testloader:
        # Move images and labels to the device
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)
    
        # Forward pass
        with torch.no_grad():
            outputs = model(X)
        
        # Get the predicted labels
        loss = torch.nn.MSELoss()(outputs, Y)
        
        # Update the total number of samples
        total_samples += 1
        
        # Update the total number of correct predictions
        total_loss += loss.item()

    # Calculate the accuracy
    loss = total_loss / total_samples

    return loss



def test_on_PACS(clients, quick):
    import torchvision.transforms as transforms
    from utils.classes import DatasetSplit
    from torch.utils.data import DataLoader
    from utils.initialization_utils import load_pacs
    import numpy as np

    client_train_datasets, val_sets, test_sets = load_pacs(path='./PACS/', BATCH_SIZE=8, nbr_clients_per_group=len(clients) // 4, augment=False)
    testset_P, testset_A, testset_C, testset_S = test_sets
    testloader_P = DataLoader(testset_P, batch_size=1, shuffle=False)
    testloader_A = DataLoader(testset_A, batch_size=1, shuffle=False)
    testloader_C = DataLoader(testset_C, batch_size=1, shuffle=False)
    testloader_S = DataLoader(testset_S, batch_size=1, shuffle=False)
    testloaders = [testloader_P, testloader_A, testloader_C, testloader_S]

    # the end goal is a 4x4 matrix with the accuracy of each group on each test set
    # acc_matrix[i,j] = a list of accuracies of group i on test set j
    acc_matrix = np.zeros((4,4,len(clients)//4))
    for i, testloader in enumerate(testloaders):
        for client in clients:
            client.best_model.to(client.device)
            j = client.group
            k = client.idx%(len(clients)//4)
            if quick:
                if i != j:
                    pass
            acc = test_model(client.best_model, testloader)
            print('Client: {} Group: {} Testset: {} Acc: {:.2f}'.format(k + j*len(clients)//4, j, i, acc))
            acc_matrix[i, j, k] = acc
            client.best_model.to('cpu')

    return acc_matrix

def test_5_clusters(clients, quick, verbose):
    import numpy as np
    # the end goal is a 5x5x20 matrix with the accuracy of each group on each test set
    # acc_matrix[i,j,k] = a list of accuracies of group i on test set j for the kth client in group i
    acc_matrix = np.zeros((5,5,len(clients)//5))

    import torchvision.transforms as transforms
    from utils.classes import DatasetSplit
    from torch.utils.data import DataLoader
    import numpy as np
    from torchvision.datasets import CIFAR10
    import time
    start = time.time()
    # --------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 test set
    testset = CIFAR10(root='.', train=False, download=True, transform=transform)

    group1_idxs = [i for i in range(len(testset)) if testset[i][1] in [0, 1]]
    group2_idxs = [i for i in range(len(testset)) if testset[i][1] in [2,3]]
    group3_idxs = [i for i in range(len(testset)) if testset[i][1] in [4,5]]
    group4_idxs = [i for i in range(len(testset)) if testset[i][1] in [6,7]]
    group5_idxs = [i for i in range(len(testset)) if testset[i][1] in [8,9]]

    # set rot deg to 0 because only label shift is implemented
    rot_deg = 0
    rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))

    testloaders = []

    for group_idxs in [group1_idxs, group2_idxs, group3_idxs, group4_idxs, group5_idxs]:
        group_set = DatasetSplit(testset,group_idxs,rot_transform)
        testloader = DataLoader(group_set, batch_size=1, shuffle=False)
        testloaders.append(testloader)

    if verbose:
        print('CIFAR-10, 5 cluster testset loaded')
        if quick:
            print('Quick test: only testing on the same group')

    for i, testloader in enumerate(testloaders):
        for client in clients:
            if verbose:
                start_client = time.time()
            if i == client.group or not quick:
                #client.best_model.to('cpu')
                client.best_model.to(client.device)
                j = client.group
                k = client.idx%(len(clients)//5)
                acc = test_model(client.best_model, testloader)
                if verbose:
                    print('Client: {} Group: {} Testset: {} Acc: {:.2f}'.format(k + j*len(clients)//5, j, i, acc))
                    print('Testing time: {:.2f} s'.format(time.time()-start_client))
                acc_matrix[i, j, k] = acc
                #client.best_model.to('cpu')
    if verbose:
        print('Testing time: {:.2f} minutes'.format((time.time()-start)/60))
    return acc_matrix

def test_on_CIFAR100(clients, quick, verbose):
    import torchvision.transforms as transforms
    from utils.classes import DatasetSplit
    from torch.utils.data import DataLoader
    import numpy as np
    from torchvision.datasets import CIFAR100
    import time
    start = time.time()

    # --------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load the CIFAR-100 test set
    testset = CIFAR100(root='.', train=False, download=True, transform=transform)

    cluster1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 75, 76, 77, 78, 79, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84])
    cluster2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 85, 86, 87, 88, 89, 50, 51, 52, 53, 54])
    cluster3 = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 45, 46, 47, 48, 49])

    idxs1 = [i for i in range(len(testset)) if testset[i][1] in cluster1]
    idxs2 = [i for i in range(len(testset)) if testset[i][1] in cluster2]
    idxs3 = [i for i in range(len(testset)) if testset[i][1] in cluster3]

    # set rot deg to 0 because only label shift is implemented
    rot_deg = 0
    rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))

    cluster1_set = DatasetSplit(testset,idxs1,rot_transform)
    testloader_cluster1 = DataLoader(cluster1_set, batch_size=1, shuffle=False)

    cluster2_set = DatasetSplit(testset,idxs2,rot_transform)
    testloader_cluster2 = DataLoader(cluster2_set, batch_size=1, shuffle=False)

    cluster3_set = DatasetSplit(testset,idxs3,rot_transform)
    testloader_cluster3 = DataLoader(cluster3_set, batch_size=1, shuffle=False)
    # --------------------------

    if verbose:
        print('CIFAR-100, label shift testset loaded')
        if quick:
            print('Quick test: only testing on the same group')

    accs = np.zeros(len(clients))
    for client in clients:
        if verbose:
            print('Testing client: {}'.format(client.idx))
            start_client = time.time()
        client.best_model.to(client.device)
        if quick:
            if client.group == 0:
                acc = test_model(client.best_model, testloader_cluster1)
            elif client.group == 1:
                acc = test_model(client.best_model, testloader_cluster2)
            elif client.group == 2:
                acc = test_model(client.best_model, testloader_cluster3)
        else:
            acc1 = test_model(client.best_model, testloader_cluster1)
            acc2 = test_model(client.best_model, testloader_cluster2)
            acc3 = test_model(client.best_model, testloader_cluster3)
            acc = [acc1, acc2, acc3]
        accs[client.idx] = acc
        # send back to cpu
        client.best_model.to('cpu')
        if verbose:
            print('Testing time: {:.2f} s'.format(time.time()-start_client))

    return accs

def test_on_CIFAR(clients, quick, verbose):
    import torchvision.transforms as transforms
    from utils.classes import DatasetSplit
    from torch.utils.data import DataLoader
    import numpy as np
    from torchvision.datasets import CIFAR10
    import time
    start = time.time()

    # --------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 test set
    testset = CIFAR10(root='.', train=False, download=True, transform=transform)

    vehicle_idxs = [i for i in range(len(testset)) if testset[i][1] in [0, 1, 8, 9]]
    animal_idxs = [i for i in range(len(testset)) if testset[i][1] in [2,3,4,5,6,7]]

    # set rot deg to 0 because only label shift is implemented
    rot_deg = 0
    rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
    vehicle_set = DatasetSplit(testset,vehicle_idxs,rot_transform)
    testloader_vehicle = DataLoader(vehicle_set, batch_size=1, shuffle=False)

    animal_set = DatasetSplit(testset,animal_idxs,rot_transform)
    testloader_animal = DataLoader(animal_set, batch_size= 1, shuffle=False)
    # --------------------------
    if verbose:
        print('CIFAR-10, label shift testset loaded')
        if quick:
            print('Quick test: only testing on the same group')

    zero_on_zero = []
    zero_on_one = []
    one_on_zero = []
    one_on_one = []

    for client in clients:
        if verbose:
            print('Testing client: {}'.format(client.idx))
            start_client = time.time()
        client.best_model.to(client.device)
        #client.best_model.to('cpu')
        if quick:
            if client.group == 0:
                acc_v = test_model(client.best_model, testloader_vehicle)
                zero_on_zero.append(acc_v)
                if verbose:
                    print('Client: {} Group: {} Acc: {:.2f}'.format(client.idx, client.group, acc_v))
            else:
                acc_a = test_model(client.best_model, testloader_animal)
                one_on_one.append(acc_a)
                if verbose:
                    print('Client: {} Group: {} Acc: {:.2f}'.format(client.idx, client.group, acc_a))
        else:
            acc_v = test_model(client.best_model, testloader_vehicle)
            acc_a = test_model(client.best_model, testloader_animal)
            if client.group == 0:
                zero_on_zero.append(acc_v)
                zero_on_one.append(acc_a)
            else:
                one_on_zero.append(acc_v)
                one_on_one.append(acc_a)
        if verbose:
            print('Testing time: {:.2f} s'.format(time.time()-start_client))
    # V_within, A_within, V_on_A, A_on_V 
    if verbose:
        print('Testing time: {:.2f} minutes'.format((time.time()-start)/60))
    
    return zero_on_zero, one_on_one, zero_on_one, one_on_zero

def test_on_toy(clients, quick = True, verbose = True):
    theta_1 = clients[0].theta
    theta_2 = clients[40].theta
    theta_3 = clients[80].theta
    sigma = 3

    n = 1000
    dataloaders = []
    for theta in [theta_1, theta_2, theta_3]:
        X, Y = generate_regression_multi(theta, n, sigma)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float().reshape(-1,1)
        dataset = torch.utils.data.TensorDataset(X,Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        dataloaders.append(dataloader)

    cluster1_losses = []
    cluster2_losses = []
    cluster3_losses = []

    for client in clients:
        if verbose:
            print('Testing client: {}'.format(client.idx))
            start_client = time.time()
        
        client.best_model.to(client.device)
        if client.group == 0:
            loss = test_toy_model(client.best_model, dataloaders[0])
            cluster1_losses.append(loss)
        elif client.group == 1:
            loss = test_toy_model(client.best_model, dataloaders[1])
            cluster2_losses.append(loss)
        elif client.group == 2:
            loss = test_toy_model(client.best_model, dataloaders[2])
            cluster3_losses.append(loss)
        if verbose:
            print('Client: {} Group: {} Loss: {:.2f}'.format(client.idx, client.group, loss))
            print('Testing time: {:.2f} s'.format(time.time()-start_client))

    return [np.array(cluster1_losses), np.array(cluster2_losses), np.array(cluster3_losses)]

def test_on_double_MNIST(clients, quick = True, verbose = True):
    import torchvision.transforms as transforms
    from utils.classes import DatasetSplit
    from torch.utils.data import DataLoader
    import numpy as np
    from torchvision.datasets import MNIST
    from torchvision.datasets import FashionMNIST
    import time

    start = time.time()

    MNIST_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    fashion_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    test_dataset_MNIST = MNIST('.', train=False, download=False, transform=MNIST_transform)
    test_dataset_fashion = FashionMNIST('.', train=False, download=False, transform=fashion_transform)

    all_idxs_MNIST = np.arange(len(test_dataset_MNIST))
    all_idxs_fashion = np.arange(len(test_dataset_fashion))

    acc_matrix = []

    from utils.classes import LabelShiftedDataset
    test_dataset_fashion = LabelShiftedDataset(test_dataset_fashion)

    ldr_MNIST = DataLoader(test_dataset_MNIST, batch_size = 1, pin_memory=False, shuffle=False)
    ldr_fashion = DataLoader(test_dataset_fashion, batch_size = 1, pin_memory=False, shuffle=False)

    for client in clients:
        if verbose:
            print('Testing client: {}'.format(client.idx))
            start_client = time.time()
        #client.best_model.to('cpu')
        client.best_model.to(client.device)
        if client.group == 0:
            acc = test_model(client.best_model, ldr_MNIST)
        elif client.group == 1:
            acc = test_model(client.best_model, ldr_fashion)
        acc_matrix.append(acc)
        if verbose:
            print('Client: {} Group: {} Acc: {:.2f}'.format(client.idx, client.group, acc))
            print('Testing time: {:.2f} s'.format(time.time()-start_client))

def test_on_fashion_MNIST(clients, quick = True, verbose = True):
    import torchvision.transforms as transforms
    from utils.classes import DatasetSplit
    from torch.utils.data import DataLoader
    import numpy as np
    from torchvision.datasets import FashionMNIST
    import time

    start = time.time()

    trans_fashion = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = FashionMNIST('.', train=False, download=True, transform=trans_fashion)

    all_idxs = np.arange(len(test_dataset))
    acc_matrix = []

    transform_0 = transforms.RandomRotation(degrees=(0,0))
    transform_0_data = DatasetSplit(test_dataset, all_idxs, transform_0)

    transform_180 = transforms.RandomRotation(degrees=(180,180))
    transform_180_data = DatasetSplit(test_dataset, all_idxs, transform_180)

    transform_10 = transforms.RandomRotation(degrees=(10,10))
    transform_10_data = DatasetSplit(test_dataset, all_idxs, transform_10)

    transform_350 = transforms.RandomRotation(degrees=(350,350))
    transform_350_data = DatasetSplit(test_dataset, all_idxs, transform_350)

    dataload_0 = DataLoader(transform_0_data, batch_size = 1, pin_memory=False, shuffle=False)
    dataload_180 = DataLoader(transform_180_data, batch_size = 1, pin_memory=False, shuffle=False)
    dataload_10 = DataLoader(transform_10_data, batch_size = 1, pin_memory=False, shuffle=False)
    dataload_350 = DataLoader(transform_350_data, batch_size = 1, pin_memory=False, shuffle=False)

    for client in clients:
        if verbose:
            print('Testing client: {}'.format(client.idx))
            start_client = time.time()
        #client.best_model.to('cpu')
        client.best_model.to(client.device)
        if client.group == 0:
            acc = test_model(client.best_model, dataload_0)
        elif client.group == 1:
            acc = test_model(client.best_model, dataload_180)
        elif client.group == 2:
            acc = test_model(client.best_model, dataload_10)
        elif client.group == 3:
            acc = test_model(client.best_model, dataload_350)
        acc_matrix.append(acc)
        if verbose:
            print('Client: {} Group: {} Acc: {:.2f}'.format(client.idx, client.group, acc))
            print('Testing time: {:.2f} s'.format(time.time()-start_client))
    
    if verbose:
        print('Testing time: {:.2f} minutes'.format((time.time()-start)/60))

    return acc_matrix
