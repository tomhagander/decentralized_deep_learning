import torch

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

def test_on_CIFAR(clients, quick, verbose):
    import torchvision.transforms as transforms
    from utils.classes import DatasetSplit
    from torch.utils.data import DataLoader
    import numpy as np
    from torchvision.datasets import CIFAR10

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
        print('CIFAR-10 testset loaded')

    zero_on_zero = []
    zero_on_one = []
    one_on_zero = []
    one_on_one = []

    for client in clients:
        if verbose:
            print('Testing client: {}'.format(client.idx))
        client.best_model.to(client.device)
        acc_v = test_model(client.best_model, testloader_vehicle)
        acc_a = test_model(client.best_model, testloader_animal)
        if client.group == 0:
            zero_on_zero.append(acc_v)
            zero_on_one.append(acc_a)
        else:
            one_on_zero.append(acc_v)
            one_on_one.append(acc_a)
    # V_within, A_within, V_on_A, A_on_V 
    return zero_on_zero, one_on_one, zero_on_one, one_on_zero