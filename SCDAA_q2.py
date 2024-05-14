import SCDAA_with_changes
import SCDAA_helper_nn
import torch.nn as nn
import torch.optim as optim
import torch
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
torch.manual_seed(0)

#torch.manual_seed(0)


T = 1
H = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64) * 0.1
M = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
sigma = 1
C = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
D = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
R = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64) * 10
spatial_grid_1D = torch.linspace(0, 1, 2)
spatial_grid = torch.tensor(list(itertools.product(spatial_grid_1D, repeat=2)), dtype = torch.float64)


LQR_steps = 10000
LQR_problem_ric = SCDAA_with_changes.LQR(H, M, C, D, R, T, sigma, LQR_steps)
sol = LQR_problem_ric.solve_ode_2()



def generate_training_data(num_samples):
    t_samples = np.random.uniform(0, 1, num_samples)
    x_samples = np.random.uniform(-3, 3, (num_samples, 2))
    return (torch.tensor(t_samples,dtype = torch.float64), torch.tensor(x_samples, dtype=torch.float64))

def train_for_value_funct():
    #num_of_epochs = 10000
    #batch_size = 640
    #num_samples = 10000
    num_of_epochs = 4000
    num_samples = 3500
    learning_rate = 0.0005
    net = SCDAA_helper_nn.Net_DGM(dim_x = 2, dim_S = 100)
    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #loss = train_for_value_funct(num_of_epochs, loss_criterion, optimizer)
    losses = []

    '''
    add testing data 
    '''
    test_samples_t, test_samples_x = generate_training_data(num_samples)
    exact_test_values = LQR_problem_ric.construct_value_funct_batch(test_samples_t, test_samples_x, sol).reshape(len(test_samples_x),1)
    test_samples_t_unsqueezed = test_samples_t.unsqueeze(1).float()
    test_samples_x_float = test_samples_x.float()
    test_errors = []

    for epoch in tqdm(range(num_of_epochs)):
        t_samples, x_samples = generate_training_data(num_samples) ## generate once --> less noise
        #permutation = torch.randperm(num_samples)        
        #for i in range(0, num_samples, batch_size):
        #   indices = permutation[i:i+batch_size]
        t_batch, x_batch = t_samples.unsqueeze(1).float(), x_samples.float()
            
        optimizer.zero_grad()
        output = net(t_batch, x_batch)

        exact = LQR_problem_ric.construct_value_funct_batch(t_batch, x_batch, sol).reshape(len(x_batch),1)

        loss = loss_criterion(output, exact)  # Exercise 1.1 loss

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        '''
        Measure the error against the training data
        '''
        test_net_out = net(test_samples_t_unsqueezed, test_samples_x_float )
        test_error = loss_criterion(test_net_out, exact_test_values) 
        test_errors.append(test_error.item())
    
    test_net_out = net(test_samples_t_unsqueezed, test_samples_x_float )
    for i in range(1,101):
        print(exact_test_values[len(test_net_out) - i - 1])
        print(test_net_out[len(test_net_out) - i - 1])
        print(test_net_out[len(test_net_out) - i - 1] - exact_test_values[len(test_net_out) - i - 1])
        print('****')
    plt.plot(np.log(test_errors))
    plt.xlabel('Epoch')
    plt.ylabel('log of Loss for the test samples')
    plt.title('Training Loss with epochs = ' + str(num_of_epochs) + ' and training data with ' + str(num_samples) + ' samples')
    plt.show()
    return()

def training_loss_for_optimal_control():
    # Define your training parameters
    num_samples = 2100
    input_size = 3  # t + x
    hidden_sizes = [100, 100]
    output_size = 2  # 2-dimensional output

    # Generate testing data
    t_test, x_test = generate_training_data(num_samples)
    input_test = torch.cat((t_test.reshape(num_samples,1), x_test), dim = -1).float()
    exact_test = LQR_problem_ric.construct_optimal_control_batch(t_test, x_test, sol)

    # Define your neural network model
    model = SCDAA_helper_nn.FFN([input_size] + hidden_sizes + [output_size])

    # Define your loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2000
    losses = []
    for epoch in tqdm(range(num_epochs)):
        t_train, x_train = generate_training_data(num_samples)
        inputs_train = torch.cat((t_train.reshape(num_samples,1), x_train), dim = -1).float()

        exact_train = LQR_problem_ric.construct_optimal_control_batch(t_train, x_train, sol)

        # Forward pass
        optimizer.zero_grad()
        outputs_train = model(inputs_train)
        loss = criterion(outputs_train, exact_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss - pass testing data into the net and measure loss against exact_test
        test_net = model(input_test)
        test_loss = criterion(test_net,exact_test)
        losses.append(test_loss.item())
        
    # Plot training loss
    plt.plot(np.log(losses))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

'''
 t, x = generate_training_data(num_samples, T)
labels = LQR_problem_ric.construct_value_funct_batch(samples[0], samples[1], sol).reshape(len(x_batch),1)
test_samples = [t,x,labels]

def compute_loss(samples):
    criterion =  nn.MSELoss()
    outputs   =  net(samples[0], samples[1])
    loss      =  criterion(outputs, samples[2])
    return loss
'''




if __name__ == '__main__':
    train_for_value_funct()
    training_loss_for_optimal_control()

#    |v(t,x) - model(t,x)|
    #input_dim = 3  # Dimension of input (t + x)
    #hidden_dim = 100  # Size of hidden layers
    #output_dim = 2  # Dimension of output
    #num_epochs = 1000
    #num_samples = 1000
    #T = 1
    #net_optimal_control = SCDAA_helper_nn.FFN([input_dim, hidden_dim, hidden_dim, output_dim])
    #optimizer = torch.optim.Adam(net_optimal_control.parameters(), lr=0.001)
    #criterion = nn.MSELoss()
    #losses = training_loss_for_optimal_control(net_optimal_control, optimizer, criterion, num_epochs, num_samples, T)
    