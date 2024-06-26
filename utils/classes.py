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
    def __init__(self, train_set=None, val_set=None, idxs_train=None, idxs_val=None, criterion=None, lr=None, 
                 device=None, batch_size=None, num_users=None, model=None, idx=None, stopping_rounds=None, 
                 ratio=None, dataset = None, shift = None, theta=None):
        self.device = device
        self.criterion = criterion
        self.lr = lr
        self.idx = idx

        self.shift = shift
        self.dataset = dataset

        self.theta = theta

        # if dataset is cifar
        if dataset == 'cifar10':
            if(shift == 'PANM_swap4'):
                if(idx<int(num_users*ratio)):
                    self.group = 0
                elif(idx<int(num_users*ratio*2)):
                    self.group = 1
                elif(idx<int(num_users*ratio*3)):
                    self.group = 2
                else:
                    self.group = 3
            elif(shift == 'PANM_swap'):
                if(idx<int(num_users*ratio)): # Only for cifar10 with 40% vehicles and 60% animals
                    self.group = 0
                else: self.group = 1
            elif(shift == 'label'):
                if(idx<int(num_users*ratio)):
                    self.group = 0
                else: self.group = 1
            elif(shift == '5_clusters'):
                if(idx<int(num_users*ratio)):
                    self.group = 0
                elif(idx<int(num_users*ratio*2)):
                    self.group = 1
                elif(idx<int(num_users*ratio*3)):
                    self.group = 2
                elif(idx<int(num_users*ratio*4)):
                    self.group = 3
                else:
                    self.group = 4

            # set rot deg to 0 because only label shift is implemented
            rot_deg = 0
            rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
            self.train_set = DatasetSplit(train_set,idxs_train,rot_transform)
            if(idxs_val):
                self.val_set = DatasetSplit(train_set,idxs_val,rot_transform)
                self.ldr_val = DataLoader(self.val_set, batch_size = 8, pin_memory=False, shuffle=False)
            
            self.ldr_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False)

        elif dataset == 'cifar100':
            if shift == 'label':
                if idx < int(num_users*0.5):
                    self.group = 0
                elif idx < int(num_users*0.75):
                    self.group = 1
                else:
                    self.group = 2
                
            rot_deg = 0
            rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
            self.train_set = DatasetSplit(train_set,idxs_train,rot_transform)
            self.val_set = DatasetSplit(train_set,idxs_val,rot_transform)
            self.ldr_val = DataLoader(self.val_set, batch_size = 8, pin_memory=False, shuffle=False)
            
            self.ldr_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False)

        elif dataset == 'PACS':
            if(idx<int(num_users*ratio)):
                self.group = 0
            elif(idx<int(num_users*ratio*2)):
                self.group = 1
            elif(idx<int(num_users*ratio*3)):
                self.group = 2
            else:
                self.group = 3

            self.train_set = train_set
            self.val_set = val_set
            
            # create train_loader, val_loader
            self.ldr_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
            self.ldr_val = DataLoader(val_set, batch_size=32, num_workers=1, pin_memory=True, shuffle=False)

        elif dataset == 'fashion_mnist':
            if idx<int(num_users*0.7):
                self.group = 0
            elif idx<int(num_users*0.9):
                self.group = 1
            elif idx<int(num_users*0.95):
                self.group = 2
            else:
                self.group = 3

            rot_degs = [0, 180, 10, 350]
            rot_deg = rot_degs[self.group]
            rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
            all_idxs = np.arange(len(train_set))
            self.train_set = DatasetSplit(train_set,all_idxs,rot_transform)
            all_idxs = np.arange(len(val_set))
            self.val_set = DatasetSplit(val_set,all_idxs,rot_transform)
            
            self.ldr_val = DataLoader(self.val_set, batch_size = 8, pin_memory=False, shuffle=False)
            self.ldr_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False)

        elif dataset == 'double':
            if idx<int(num_users*0.5):
                self.group = 0
            else:
                self.group = 1

            rot_transform = transforms.RandomRotation(degrees=(0,0))
            all_idxs = np.arange(len(train_set))
            self.train_set = DatasetSplit(train_set,all_idxs,rot_transform)
            all_idxs = np.arange(len(val_set))
            self.val_set = DatasetSplit(val_set,all_idxs,rot_transform)
            
            self.ldr_val = DataLoader(self.val_set, batch_size = 8, pin_memory=False, shuffle=False)
            self.ldr_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False)

        elif dataset == 'toy_problem':
            if idx < int(num_users/3):
                self.group = 0
            elif idx < int(2*num_users/3):
                self.group = 1
            else:
                self.group = 2

            self.train_set = train_set
            self.val_set = val_set
            self.ldr_val = DataLoader(val_set, batch_size = 8, shuffle=True)
            self.ldr_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        
        # copy model and send to device
        self.local_model = copy.deepcopy(model)
        if dataset != 'cifar100':
            self.local_model.to(self.device) # change 2

        # Early stopping
        self.early_stopping = EarlyStopping(patience=stopping_rounds, min_delta=0)
        
        # place to store best model
        self.best_model = copy.deepcopy(self.local_model)
        
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_acc_list = []

        self.val_losses_post_exchange = []
        self.val_accs_post_exchange = []

        self.n_sampled = np.zeros(num_users, dtype=int)
        self.n_sampled_prev = np.zeros(num_users)
        self.n_selected = np.zeros(num_users)
        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf
        self.count = 0
        self.all_similarities = []
        self.exchanges_every_round = []

        # validate initial model
        if dataset == 'toy_problem':
            init_loss = self.toy_validate(self.local_model, train_set = False)
            self.val_loss_list.append(init_loss)
            init_train_loss = self.toy_validate(self.local_model, train_set = True)

        else:
            init_loss, init_acc = self.validate(self.local_model, train_set = False)
            self.val_loss_list.append(init_loss)
            self.val_acc_list.append(init_acc)
            init_train_loss, init_train_acc = self.validate(self.local_model, train_set = True)
        self.train_loss_list.append(init_train_loss)

        # gradients (for cosine similarity)
        self.grad_a = None
        self.grad_b = None
        self.initial_weights = copy.deepcopy(self.local_model.state_dict())
        self.last_weights = None #copy.deepcopy(self.local_model.state_dict())

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

        # true_similarities
        self.true_similarities = []

        # taus for fixed entropy
        self.all_taus = []

        # for mergatron
        self.pre_merge_weights = None # copy.deepcopy(self.local_model.state_dict())
        
        
    def train(self,n_epochs):
        # save last weights
        self.last_weights = None #copy.deepcopy(self.local_model.state_dict())

        if self.dataset == 'cifar100':
            self.local_model.to(self.device) # change 1

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
        
        if self.dataset == 'cifar100':
            del optimizer
            self.local_model.to('cpu')
            
        return self.best_model, epoch_loss[-1], self.best_val_loss, self.best_val_acc
    
    def toy_train(self, n_epochs):
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr)
        self.local_model.train()
        train_loss = []
        for iter in range(n_epochs):
            epoch_loss = 0
            epoch_acc = 0
            for batch_idx, (x, y) in enumerate(self.ldr_train):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.local_model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                #epoch_acc += (outputs.argmax(1) == y).sum().item()
            train_loss.append(epoch_loss/len(self.ldr_train))
        
        self.train_loss_list.append(train_loss[-1])
        # validate
        val_loss = self.toy_validate(self.local_model, train_set = False)
        self.val_loss_list.append(val_loss)

        if(val_loss < self.best_val_loss):
            self.count = 0
            self.best_val_loss = val_loss
            self.best_model.load_state_dict(self.local_model.state_dict())

        # early stopping
        self.early_stopping(val_loss)

        # could be minus one
        return self.local_model, train_loss
    
    def toy_validate(self, model, train_set):
        if(train_set):
            ldr = self.ldr_train
        else:
            ldr = self.ldr_val
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in ldr:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                test_loss += self.criterion(outputs, y).item()

        return test_loss/len(ldr)

    
    def validate(self,model,train_set):
        if(train_set):
            ldr = self.ldr_train
        else:
            ldr = self.ldr_val
        
        if self.dataset == 'cifar100':
            model.to(self.device)
        # model.to(self.device) # change 1
        
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

        if self.dataset == 'cifar100':
            model.to('cpu')
        # model.to('cpu') # change 1
        return val_loss, val_acc
    
    def get_grad_a(self):
        # find difference between current weights and last weights
        w_diff = {}
        for (name, param) in self.local_model.named_parameters():
            w_diff[name] = param - self.last_weights[name]
        all_gradients = []
        for p in w_diff.values(): 
            grad = p.view(-1).cpu().detach().numpy()
            all_gradients.append(grad)
        return np.concatenate(all_gradients)
    
        '''
        all_gradients = []
        for p in self.local_model.parameters():
            grad = p.grad.view(-1).cpu().detach().numpy()
            all_gradients.append(grad)
        return np.concatenate(all_gradients)
        # implementation from paper might be better here as method would be same as for b
        '''
    
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
    
    def get_grad_origin(self):
        # find difference between current weights and origin
        w_diff = {}
        for (name, param) in self.local_model.named_parameters():
            w_diff[name] = param
        all_gradients = []
        for p in w_diff.values(): 
            grad = p.view(-1).cpu().detach().numpy()
            all_gradients.append(grad)
        return np.concatenate(all_gradients)
    
    def measure_all_similarities(self, all_clients, similarity_metric, alpha=0, store=True):
        if self.dataset == 'cifar10':
            if self.shift == 'label':
                if self.idx != 0 and self.idx != 70:
                    return np.zeros(len(all_clients))
            elif self.shift == '5_clusters':
                if self.idx != 0 and self.idx != 20 and self.idx != 40 and self.idx != 60 and self.idx != 80:
                    return np.zeros(len(all_clients))
        elif self.dataset == 'fashion_mnist':
            if self.idx != 0 and self.idx != 70 and self.idx != 90 and self.idx != 95:
                return np.zeros(len(all_clients))
        elif self.dataset == 'double':
            if self.idx != 0 and self.idx != 50:
                return np.zeros(len(all_clients))
        elif self.dataset == 'toy_problem':
            if self.idx != 0 and self.idx != 40 and self.idx != 80:
                return np.zeros(len(all_clients))
        elif self.dataset == 'cifar100':
            if self.shift == 'label':
                if self.idx != 0 and self.idx != 30 and self.idx != 50:
                    return np.zeros(len(all_clients))
            
        print('Measuring similarities of client {}'.format(self.idx))
        similarities = np.zeros(len(all_clients))
        for client in all_clients:
            if client.idx != self.idx:
                if client.early_stopping.is_stopped() and self.early_stopping.is_stopped(): # test if early stopping is stopped
                    similarities[client.idx] = self.true_similarities[-1][client.idx]
                elif similarity_metric == 'cosine_similarity':
                    # measure cosine similarity between all clients
                    #self_grad_a = self.get_grad_a() # a type similarity not used now to speed up
                    self_grad_b = self.get_grad_b()
                    #client_grad_a = client.get_grad_a()
                    client_grad_b = client.get_grad_b()
                    # get cosine similarity from torch function cosine_similarity
                    #cosine_similarity_a = torch.nn.functional.cosine_similarity(torch.tensor(self_grad_a), torch.tensor(client_grad_a), dim=0)
                    cosine_similarity_b = torch.nn.functional.cosine_similarity(torch.tensor(self_grad_b), torch.tensor(client_grad_b), dim=0)
                    # calculate cosine similarity
                    #cosine_similarity = alpha*cosine_similarity_a + (1-alpha)*cosine_similarity_b
                    similarities[client.idx] = cosine_similarity_b
                elif similarity_metric == 'inverse_training_loss':
                    # take clients model and validate it on own training set
                    client_model = copy.deepcopy(client.local_model)
                    val_loss, _ = self.validate(client_model, train_set = True)
                    similarities[client.idx] = 1/(val_loss + 1e-6)

                elif similarity_metric == 'cosine_origin':
                    self_grad_origin = self.get_grad_origin()
                    client_grad_origin = client.get_grad_origin()
                    # get cosine similarity from torch function cosine_similarity
                    cosine_similarity_origin = torch.nn.functional.cosine_similarity(torch.tensor(self_grad_origin), torch.tensor(client_grad_origin), dim=0)
                    # calculate cosine similarity
                    similarities[client.idx] = cosine_similarity_origin

                elif similarity_metric == 'l2':
                    self_grad_origin = self.get_grad_origin()
                    client_grad_origin = client.get_grad_origin()
                    # get cosine similarity from torch function cosine_similarity
                    l2_distance = np.linalg.norm(self_grad_origin - client_grad_origin)
                    # calculate cosine similarity
                    similarities[client.idx] = 1/(l2_distance + 1e-6)
                else:
                    pass 
        
        if store:
            self.true_similarities.append(similarities)
            
        return similarities


# Custom dataset to shift labels
class LabelShiftedDataset(Dataset):
    def __init__(self, dataset, label_shift=10):
        self.dataset = dataset
        self.label_shift = label_shift

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label += self.label_shift
        return image, label