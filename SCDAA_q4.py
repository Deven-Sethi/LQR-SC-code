import SCDAA_with_changes
import SCDAA_helper_nn
import torch
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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

d = 2 
hidden_dim = 100
num_samples = 500
num_of_epochs = 2000
learning_rate = 0.001

# test_grid 
no_of_test_samples = 50
test_times = torch.tensor(np.random.uniform(0, 1, no_of_test_samples), dtype = torch.float64)
test_space = torch.tensor(np.random.uniform(-3, 3, (no_of_test_samples, 2)), dtype = torch.float64)
exact_optimal_control_list = torch.zeros((no_of_test_samples,2), dtype = torch.float64)
exact_value_funct_list = torch.zeros(no_of_test_samples, dtype = torch.float64)
for i in range(no_of_test_samples):
    exact_optimal_control_list[i] = LQR_problem_ric.construct_optimal_control(test_times[i].item(),test_space[i],sol)
    exact_value_funct_list[i] = LQR_problem_ric.construct_value_funct(test_times[i].item(),test_space[i],sol)

# exact values
exact_optimal_control = []
exact_value_funct = []

losses_optimal_control = []
losses_value_function = []

current_control = torch.zeros(num_samples, 2)
current_control[:]=torch.tensor([1,1], dtype = torch.float64)
PIA_approximation = SCDAA_helper_nn.policy_iteration_training(d = 2, hidden_dim = 100, num_samples = num_samples, 
                                                              num_of_epochs = num_of_epochs, sigma = sigma, 
                                                              H = H, M = M, C = C, D = D, T = T, R = R,
                                                              lr = learning_rate)
current_approx_value_funct = PIA_approximation.train_initial_cost_functional_for_const_control(current_control)
num_of_iterations = 1500
pbar_pia = tqdm(num_of_iterations)

# reshape the test grids
t_test = (test_times.reshape(no_of_test_samples,1)).float()
x_test = test_space.float()
exact_value_funct_list_reshape = exact_value_funct_list.reshape(no_of_test_samples,1)
for i in range(num_of_iterations):
    new_control = PIA_approximation.learning_next_control(current_approx_value_funct)
    
    error_control = abs(new_control(torch.cat((t_test, x_test), dim = -1)) - exact_optimal_control_list)
    losses_optimal_control.append(torch.mean(error_control / abs(exact_optimal_control_list)).item())
    
    current_approx_value_funct = PIA_approximation.learning_next_value_function(new_control)
    losses_value_function.append((torch.mean(abs(current_approx_value_funct(t_test,x_test)-exact_value_funct_list_reshape) / exact_value_funct_list_reshape)).item())
    
    pbar_pia.update(1)

    if i % 5 == 0:
        print(i,losses_optimal_control[-1],losses_value_function[-1])

torch.save(new_control.state_dict(), 'trained_optimal_control.pth')
torch.save(current_approx_value_funct.state_dict(), 'trained_value_funct.pth')

# plot a contour for value function
spatial_grid_1D = torch.linspace(-3, 3, 50)
spatial_grid = torch.tensor(list(itertools.product(spatial_grid_1D, repeat=2)), dtype = torch.float64)
for t in torch.linspace(0, 1, 5, dtype = torch.float64):
    time_stretched = (t.repeat(len(spatial_grid))).reshape(len(spatial_grid), 1)
    x_vals = spatial_grid_1D.reshape(len(spatial_grid_1D),1)
    y_vals = spatial_grid_1D.reshape(len(spatial_grid_1D),1)
    z_vals = current_approx_value_funct(time_stretched.float(), spatial_grid.float())
    z_vals_no_grad = z_vals.detach()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x_vals, y_vals, z_vals_no_grad.reshape(len(x_vals), len(y_vals)), cmap='viridis')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('v(t,x) ' + ' for t = ' + str(t.item()))    
    ax.set_title('Plot for the Value Function For t = ' + str(t.item()))
    fig.colorbar(surface, shrink=0.5, aspect=5)
    plt.savefig("time_equal_to_" + str(t.item()) + "_value_function.pdf")
    plt.close()
    

x_values = np.arange(len(losses_optimal_control))
plt.plot(x_values,np.log(losses_optimal_control))
plt.xlabel('Epoch')
plt.ylabel('Natual Log of the Mean Relative Error Squared For Optimal Control')
plt.title('Training Loss with epochs = ' + str(num_of_epochs) + ' and testing data with containing' + str(no_of_test_samples) + ' samples')
plt.savefig("Training Loss_optimal_control.pdf")
plt.close()
plt.show()


x_values = np.arange(len(losses_optimal_control))
plt.plot(x_values,np.log(losses_value_function))
plt.xlabel('Epoch')
plt.ylabel('Natual Log of the Mean Relative Error Squared For Value Function')
plt.title('Training Loss with epochs = ' + str(num_of_epochs) + ' and testing data with containing' + str(no_of_test_samples) + ' samples')
plt.savefig("Training Loss_value_funct.pdf")
plt.close()
plt.show()