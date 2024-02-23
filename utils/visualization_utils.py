import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_client_validation_accuracy(client, figpath):
    plt.figure()
    plt.plot(client.val_acc_list)
    plt.title('Validation accuracy for client {}'.format(client.idx))
    plt.show()
    # save figure
    plt.savefig(figpath+'validation_accuracy_client_{}.png'.format(client.idx))

def plot_client_validation_and_training_loss(client, figpath):
    plt.figure()
    plt.plot(client.train_loss_list)
    plt.plot(client.val_loss_list)
    # add legend
    plt.legend(['train', 'val'], loc='upper right')
    plt.title('Training and validation loss for client {}'.format(client.idx))
    plt.show()
    # save figure
    plt.savefig(figpath+'validation_and_training_loss_client_{}.png'.format(client.idx))

def plot_average_client_validation_accuracy(clients, figpath):
    plt.figure()
    avg_acc = np.mean([client.val_acc_list for client in clients], axis=0)
    plt.plot(avg_acc)
    plt.title('Average validation accuracy')
    plt.show()
    # save figure
    plt.savefig(figpath+'average_validation_accuracy.png')

def plot_average_client_validation_and_training_loss(clients, figpath):
    # plot training and validation loss
    plt.figure()
    avg_loss = np.mean([client.train_loss_list for client in clients], axis=0)
    avg_loss1 = np.mean([client.val_loss_list for client in clients], axis=0)
    plt.plot(avg_loss)
    plt.plot(avg_loss1)
    plt.title('Average training and validation loss')
    # add legend
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    # save figure
    plt.savefig(figpath+'average_validation_and_training_loss.png')

def plot_client_similarity_scores(client, figpath):
    plt.figure()
    plt.plot(client.similarity_scores)
    plt.title('Final similarity scores')
    plt.show()
    # save figure
    plt.savefig(figpath+'similarity_scores_client_{}.png'.format(client.idx))

def plot_similarity_heatmap(clients, figpath):
    plt.figure()
    similarity_matrix = np.zeros((len(clients), len(clients)))
    for i, client in enumerate(clients):
        similarity_matrix[i] = client.similarity_scores
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.title('Similarity heatmap')
    # add colorbar
    plt.colorbar()
    # save figure
    plt.show()
    plt.savefig(figpath+'similarity_heatmap.png')

def plot_priors_heatmap(clients, figpath):
    plt.figure()
    priors_matrix = np.zeros((len(clients), len(clients)))
    for i, client in enumerate(clients):
        priors_matrix[i] = client.priors
    plt.imshow(priors_matrix, cmap='hot', interpolation='nearest')
    plt.title('Priors heatmap')
    # add colorbar
    plt.colorbar()
    # save figure
    plt.show()
    plt.savefig(figpath+'priors_heatmap.png')

def plot_neighbor_sampled_heatmap(clients, figpath):
    plt.figure()
    n_sampled_matrix = np.zeros((len(clients), len(clients)))
    for i, client in enumerate(clients):
        n_sampled_matrix[i] = client.n_sampled
    plt.imshow(n_sampled_matrix, cmap='hot', interpolation='nearest')
    plt.title('Number of times sampled heatmap')
    # add colorbar
    plt.colorbar()
    # save figure
    plt.show()
    plt.savefig(figpath+'n_sampled_heatmap.png')

def plot_variable_nbr_of_clients_validation_accuracy(clients, nbr, figpath):
    plt.figure()
    # pick nbr clients randomly
    clients_picked = np.random.choice(clients, nbr)
    for client in clients_picked:
        plt.plot(client.val_acc_list)
    plt.title('Validation accuracy for clients')
    plt.show()
    # save figure
    plt.savefig(figpath+'val_acc_clients_picked.png')


def similarity_landscape(client, figpath):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for x and y coordinates
    x = np.arange(len(client.all_similarities[1]))
    y = np.arange(len(client.all_similarities[0]))
    X, Y = np.meshgrid(x, y)

    # Plot the surface
    ax.plot_surface(X, Y, client.all_similarities, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Distribution')
    ax.set_zlabel('Similarity Metric')
    ax.set_title('Similarity Metric Distributions over Time')

    # show colorbar
    plt.colorbar(ax.plot_surface(X, Y, client.all_similarities, cmap='viridis'))

    # Show the plot
    plt.show()

    # save figure
    plt.savefig(figpath+'similarity_landscape.png')

    # save other rotation angles
    ax.view_init(30, 45)
    plt.show()
    plt.savefig(figpath+'similarity_landscape_2.png')

    ax.view_init(30, 135)
    plt.show()
    plt.savefig(figpath+'similarity_landscape_3.png')

    ax.view_init(30, 225)
    plt.show()
    plt.savefig(figpath+'similarity_landscape_4.png')

    ax.view_init(30, 315)
    plt.show()
    plt.savefig(figpath+'similarity_landscape_5.png')