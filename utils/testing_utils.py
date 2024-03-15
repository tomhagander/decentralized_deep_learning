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