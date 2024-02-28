import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

# not entirely sure what this is for
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, transform):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform:
            image = self.transform(image)
        return image, label
    

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        '''
        Initializes the EarlyStopping instance.
        :param patience: Number of epochs to wait after min has been hit.
        :param min_delta: Minimum change to qualify as an improvement.
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_score = None
        self.stop_training = False

    def __call__(self, val_loss):
        '''
        Evaluates the current validation loss.
        :param val_loss: Current validation loss.
        :return: True if the training should stop, False otherwise.
        '''
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.stop_training = True
        return self.stop_training
    
    def is_stopped(self):
        '''
        Returns whether the training should stop or not.
        :return: True if the training should stop, False otherwise.
        '''
        return self.stop_training

    


# client object
class Client(object):
    def __init__(self, train_set, idxs_train, idxs_val, criterion, lr, device, batch_size, num_users, model, idx, stopping_rounds):
        self.device = device
        self.criterion = criterion
        self.lr = lr
        self.idx = idx
        if(idx<int(num_users*0.4)): # Only for cifar10 with 40% vehicles and 60% animals
            self.group = 0
        else: self.group = 1

        # set rot deg to 0 because only label shift is implemented
        rot_deg = 0
        rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
        self.train_set = DatasetSplit(train_set,idxs_train,rot_transform)
        if(idxs_val):
            self.val_set = DatasetSplit(train_set,idxs_val,rot_transform)
            self.ldr_val = DataLoader(self.val_set, batch_size = 1, shuffle=False)
        
        self.ldr_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        
        # copy model and send to device
        self.local_model = copy.deepcopy(model)
        self.local_model.to(self.device)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=stopping_rounds, min_delta=0)
        
        # place to store best model
        self.best_model = copy.deepcopy(self.local_model)
        
        self.received_models = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_acc_list = []

        self.val_losses_post_exchange = []
        self.val_accs_post_exchange = []

        self.n_received = 0
        self.n_sampled = np.zeros(num_users, dtype=int)
        self.n_sampled_prev = np.zeros(num_users)
        self.n_selected = np.zeros(num_users)
        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf
        self.count = 0
        self.all_similarities = []

        # validate initial model
        init_loss, init_acc = self.validate(self.local_model, train_set = False)
        self.val_loss_list.append(init_loss)
        self.val_acc_list.append(init_acc)
        init_train_loss, init_train_acc = self.validate(self.local_model, train_set = True)
        self.train_loss_list.append(init_train_loss)

        # gradients (for cosine similarity)
        self.grad_a = None
        self.grad_b = None
        self.initial_weights = copy.deepcopy(self.local_model.state_dict())

        # more stuff for cosine similarity
        self.N = []
        self.B = []

        # prior distribution over neighbors, this can be cleaned up
        # uniform distribution
        self.priors = np.ones(num_users)/(num_users-1)
        self.priors[idx] = 0.0

        # might need 
        self.similarity_scores = np.zeros(num_users)
        self.neighbour_list = []
        
    def train(self,n_epochs):
        self.local_model.train()
        optimizer = torch.optim.Adam(self.local_model.parameters(),lr=self.lr)
        
        epoch_loss = []
        
        for iter in range(n_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = self.local_model(images.float())
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        self.grad_a = self.get_grad_a()
        self.grad_b = self.get_grad_b()
        
        self.train_loss_list.append(epoch_loss[-1])
        val_loss, val_acc = self.validate(self.local_model, train_set = False)
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
        
        if(val_loss < self.best_val_loss):
            self.count = 0
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_model.load_state_dict(self.local_model.state_dict())
        else:
            self.count += 1

        # early stopping
        self.early_stopping(val_loss)
            
        return self.best_model, epoch_loss[-1], self.best_val_loss, self.best_val_acc
    
    def validate(self,model,train_set):
        if(train_set):
            ldr = self.ldr_train
        else:
            ldr = self.ldr_val
            
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(ldr):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                log_probs = model(inputs)
                _, predicted = torch.max(log_probs.data, 1)
                                         
                loss = self.criterion(log_probs,labels)                
                batch_loss.append(loss.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            val_loss = sum(batch_loss)/len(batch_loss)

        return val_loss, val_acc
    
    def get_grad_a(self):
        all_gradients = []
        for p in self.local_model.parameters():
            grad = p.grad.view(-1).cpu().detach().numpy()
            all_gradients.append(grad)
        return np.concatenate(all_gradients)
        # implementation from paper might be better here as method would be same as for b
    
    def get_grad_b(self):
        # find difference between current weights and initial weights
        w_diff = {}
        for (name, param) in self.local_model.named_parameters():
            w_diff[name] = param - self.initial_weights[name]
        all_gradients = []
        for p in w_diff.values(): 
            grad = p.view(-1).cpu().detach().numpy()
            all_gradients.append(grad)
        return np.concatenate(all_gradients)