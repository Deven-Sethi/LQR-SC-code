import SCDAA_with_changes
import SCDAA_helper_nn
import torch.nn as nn
import torch.optim as optim
import torch
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def generate_training_data(num_samples):
    t_samples = np.random.uniform(0, 1, num_samples)
    x_samples = np.random.uniform(-3, 3, (num_samples, 2))
    return (torch.tensor(t_samples,dtype = torch.float64), torch.tensor(x_samples, dtype=torch.float64))

def batch_Mat_batch(batch_1, Mat, batch_2): 
    N = len(batch_1)
    Mat_batch = torch.zeros(N, 2,2)
    Mat_batch[:] = Mat
    batch_1_reshaped = batch_1.unsqueeze(-1)
    result = torch.matmul(Mat_batch, batch_1_reshaped).squeeze(-1)
    return(torch.sum(result * batch_2, axis = 1).reshape(N, 1))

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
#sol = LQR_problem_ric.solve_ode_2()
d = 2 
hidden_dim = 100
num_samples = 500
num_of_epochs = 3000
learning_rate = 0.001
gen_new_samples = 50


test_no_of_samples = 2
test_samples_t, test_samples_x = generate_training_data(test_no_of_samples)
print(test_samples_t, test_samples_x)
test_exact_v_tx = torch.zeros(test_no_of_samples)
MC = 50000

terminal_time_grid = torch.tensor([1]).repeat(test_no_of_samples,1)
for i in range(test_no_of_samples):
    test_exact_v_tx[i] = LQR_problem_ric.MC_approximation_constant_control(test_samples_t[i].item(), test_samples_x[i], MC)
            
test_samples_t_reshape = test_samples_t.reshape(test_no_of_samples, 1).float()
test_samples_x_reshape = test_samples_x.float()
test_exact_v_tx_reshape = test_exact_v_tx.reshape(test_no_of_samples, 1).float()
#test_exact_v_tx_terminal_reshape = test_exact_v_tx_terminal.reshape(test_no_of_samples,1)
#print(test_exact_v_tx_terminal_reshape)
print(test_exact_v_tx_reshape)
        


train_PDE = SCDAA_helper_nn.linear_parabolic_PDE_solver(d = 2, 
                                                        hidden_dim = 100, 
                                                        num_samples = num_samples, 
                                                        num_of_epochs = num_of_epochs,  
                                                        sigma = sigma,
                                                        H = H, 
                                                        M = M, 
                                                        C = C, 
                                                        D = D, 
                                                        T = T, 
                                                        R = R,
                                                        lr = learning_rate,
                                                        gen_samples = gen_new_samples)


cost_control = torch.zeros(num_samples, 2)
cost_control[:]=torch.tensor([1,1], dtype = torch.float64)
losses = train_PDE.train(test_samples_t_reshape, test_samples_x_reshape, test_exact_v_tx_reshape, cost_control)

print(num_samples, num_of_epochs, learning_rate, gen_new_samples)
print(H, M, C, D, R, T, sigma)
plt.plot(np.log(losses[0]))
plt.xlabel('Epoch')
plt.ylabel('Natual Log of the Mean Relative Error Squared')
plt.title('Training Loss with epochs = ' + str(num_of_epochs) + ' and training data with ' + str(num_samples) + ' samples')
plt.show()

# could also check the optimal control


