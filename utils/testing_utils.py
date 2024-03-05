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
