import torch 
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


def random_seeding(use_cuda = True):
    random.seed(123)
    np.random.seed(123) # cpu vars
    torch.manual_seed(123) # cpu  vars
    if use_cuda: torch.cuda.manual_seed_all(123) # gpu vars


def get_labels(X_seq, victim_model):
    y_seq = victim_model.predict(X_seq)
    return y_seq


def naive_trainer(model, train_loader, val_loader, optimizer = None, n_epochs = 500):
    """
    make sure to run random_seeding before INITIALIZING model
    or better, when intializing model give net = Net(seed)
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(weight_decay = 0.001) # added only wt decay
    criterion = nn.CrossEntropyLoss() # too many options is bad
      
    for epoch in range(n_epochs):
        print(f'\nepoch {epoch}')
        for (inputs, labels) in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs).squeeze()
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            print(f"\rloss: {loss.item()}", end = "", flush = True)

    with torch.no_grad():
        avg_val_loss = 0.
        for (inputs_val, y_val) in val_loader:
            inputs_val, y_val = inputs_val.to(device), y_val.to(device)
            val_preds = model(inputs_val).squeeze()
            val_loss = criterion(val_preds, y_val)
            avg_val_loss += val_loss.item() 

    print('\n\nFinished Training')
    return avg_val_loss / len(val_loader.dataset) # return final val loss


def improved_trainer(model, train_loader, val_loader, optimizer = None, 
                     n_epochs = 500, avg_last = 5): # set good default for N
    """
    make sure to run random_seeding before INITIALIZING model
    or better, when intializing model give net = Net(seed)
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(weight_decay = 0.001) # added only wt decay
    criterion = nn.CrossEntropyLoss() # too many options is bad

    val_loss_dict = dict()
    val_loss_list = deque(maxlen = avg_last) # can be slow if avg_last is large

    for epoch in range(n_epochs):
        print(f'\nepoch {epoch}')
        for (inputs, labels) in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs).squeeze()
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            print(f"\rloss: {loss.item()}", end = "", flush = True)

        with torch.no_grad():
            avg_val_loss = 0.
            for (inputs_val, y_val) in val_loader:
                inputs_val, y_val = inputs_val.to(device), y_val.to(device)
                val_preds = model(inputs_val).squeeze()
                val_loss = criterion(val_preds, y_val)
                avg_val_loss += val_loss.item()

        val_loss_list.append(avg_val_loss)  # avg the last 'avg_last' epochs
        val_loss_dict[epoch] = np.mean(val_loss_list) / len(val_loader.dataset)

        stop_condition = None # define stop condition
        if stop_condition:
            break

    print('\n\nFinished Training')
    return val_loss_dict

