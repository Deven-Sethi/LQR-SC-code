import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import SCDAA_with_changes
#

class DGM_Layer(nn.Module):
    '''
    This code is copied from Marc's GitHub
    https://github.com/msabvid/Deep-PDE-Solvers/blob/master/lib/dgm.py
    '''
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))
            

        self.gate_Z = self.layer(dim_x+dim_S, dim_S)
        self.gate_G = self.layer(dim_x+dim_S, dim_S)
        self.gate_R = self.layer(dim_x+dim_S, dim_S)
        self.gate_H = self.layer(dim_x+dim_S, dim_S)
            
    def layer(self, nIn, nOut):
        l = nn.Sequential(nn.Linear(nIn, nOut), self.activation)
        return l
    
    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        
        input_gate_H = torch.cat([x, S*R],1)
        H = self.gate_H(input_gate_H)
        
        output = ((1-G))*H + Z*S
        return output

class Net_DGM(nn.Module):
    '''
    This code is copied from Marc's GitHub
    https://github.com/msabvid/Deep-PDE-Solvers/blob/master/lib/dgm.py
    '''
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()

        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S), self.activation)

        self.DGM1 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM2 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM3 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)

        self.output_layer = nn.Linear(dim_S, 1)

    def forward(self,t,x):
        tx = torch.cat([t,x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        output = self.output_layer(S4)
        return output
    
    
class FFN(nn.Module):
    # size = [dimX + 1] + [100, 100] + [dimX]

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False):
        super().__init__()
        
        layers = [nn.BatchNorm1d(sizes[0]),] if batch_norm else []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j+1], affine=True))
            if j<(len(sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True

    def forward(self, x):
        return self.net(x)
    
def get_gradient(output, x):
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

def get_laplacian(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess_diag.append(grad2[:,d].view(-1,1))    
    hess_diag = torch.cat(hess_diag,1)
    laplacian = hess_diag.sum(1, keepdim=True)
    return laplacian

class PDE_DGM_BlackScholes(nn.Module):
    '''x_T_C_x = batch_Mat_batch(x_batch, self.C, x_batch)
    This code is from Marc's Git-Hub - should be used as an examples of how to solve parabolic pdes
    using learning algorithms
    '''
    def __init__(self, d: int, hidden_dim: int, mu:float, sigma: float, ts: torch.Tensor=None):

        super().__init__()
        self.d = d
        self.mu = mu
        self.sigma = sigma

        self.net_dgm = Net_DGM(d, hidden_dim, activation='Tanh')
        self.ts = ts

    def fit(self, max_updates: int, batch_size: int, option, device):

        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)
        loss_fn = nn.MSELoss()
        
        pbar = tqdm.tqdm(total=max_updates)

        torch.manual_seed(0)

        for it in range(max_updates):
            optimizer.zero_grad()

            input_domain = 0.5 + 2*torch.rand(batch_size, self.d, device=device, requires_grad=True)
            t0, T = self.ts[0], self.ts[-1]
            t = t0 + T*torch.rand(batch_size, 1, device=device, requires_grad=True)
            u_of_tx = self.net_dgm(t, input_domain)
            grad_u_x = get_gradient(u_of_tx,input_domain)
            grad_u_t = get_gradient(u_of_tx, t)
            laplacian = get_laplacian(grad_u_x, input_domain)
            target_functional = torch.zeros_like(u_of_tx)
            pde = grad_u_t + torch.sum(self.mu*input_domain.detach()*grad_u_x,1, keepdim=True) + 0.5*self.sigma**2 * laplacian - self.mu * u_of_tx
            MSE_functional = loss_fn(pde, target_functional)
            
            input_terminal = 0.5 + 2*torch.rand(batch_size, self.d, device=device, requires_grad=True)
            t = torch.ones(batch_size, 1, device=device) * T
            u_of_tx = self.net_dgm(t, input_terminal)
            target_terminal = option.payoff(input_terminal)
            MSE_terminal = loss_fn(u_of_tx, target_terminal)

            loss = MSE_functional + MSE_terminal
            loss.backward()
            optimizer.step()
            scheduler.step()
            if it%10 == 0:
                pbar.update(10)
                pbar.write("Iteration: {}/{}\t MSE functional: {:.4f}\t MSE terminal: {:.4f}\t Total Loss: {:.4f}".format(it, max_updates, MSE_functional.item(), MSE_terminal.item(), loss.item()))


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

class linear_parabolic_PDE_solver(nn.Module):
    def __init__(self, d, hidden_dim, num_samples, num_of_epochs, sigma, H, M, C, D, T, R, lr,gen_samples):
        super().__init__()
        self.dim = d # dimension of input data 
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples # number of samples we want to use to train our model
        self.num_of_epochs = num_of_epochs

        self.net_dgm = Net_DGM(dim_x = self.dim, dim_S = self.hidden_dim, activation='Tanh')

        self.sigma = sigma
        self.H = H
        self.M = M 
        self.C = C
        self.D = D
        self.T = T
        self.R = R
        self.lr = lr
        self.gen_samples = gen_samples


        self.LQR_steps = 10000
        self.LQR_problem = SCDAA_with_changes.LQR(self.H, self.M, self.C, self.D, self.R, self.T, self.sigma, self.LQR_steps) 


    def train(self, test_samples_t, test_samples_x, test_exact_v_tx, alpha):
        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)
        loss_fn = nn.MSELoss() 

        pbar = tqdm(total=self.num_of_epochs)

        losses_interior = []
        #losses_teminal = []
        
        test_approximation = None
        #test_approximation_terminal = None
        #train_samples_t, train_samples_x = generate_training_data(self.num_samples)
        #t_batch, x_batch = train_samples_t.unsqueeze(1).float(), train_samples_x.float()
        #t_batch.requires_grad = True
        #x_batch.requires_grad = True
        model = self.net_dgm
        for epoch in range(self.num_of_epochs):
            optimizer.zero_grad()

            # generate the training data 
            #if epoch % self.gen_samples == 0:
            train_samples_t, train_samples_x = generate_training_data(self.num_samples)
            t_batch, x_batch = train_samples_t.unsqueeze(1).float(), train_samples_x.float()
            t_batch.requires_grad = True
            x_batch.requires_grad = True
            
            #train_u_tx = self.net_dgm(t_batch, x_batch) # the current approximation of a cost functional with current 
            train_u_tx = model(t_batch, x_batch)
            grad_u_x = get_gradient(train_u_tx,x_batch)
            grad_u_t = get_gradient(train_u_tx, t_batch)
            laplacian = get_laplacian(grad_u_x, x_batch)
            target_functional = torch.zeros_like(train_u_tx)
            grad_u_x_H_x = batch_Mat_batch(grad_u_x, self.H, x_batch)
            grad_u_x_M_alpha = batch_Mat_batch(grad_u_x, self.M, alpha)
            x_T_C_x = batch_Mat_batch(x_batch, self.C, x_batch)
            alpha_D_alpha = batch_Mat_batch(alpha, self.D, alpha)
            pde = grad_u_t + 0.5 * self.sigma ** 2 * laplacian + grad_u_x_H_x + grad_u_x_M_alpha + x_T_C_x + alpha_D_alpha
            MSE_functional = loss_fn(pde, target_functional)

            '''
            Now do the terminal condition 
            '''
            t = torch.ones(self.num_samples, 1) * self.T
            input_terminal = generate_training_data(self.num_samples)[1].float()
            exact_terminal = batch_Mat_batch(input_terminal, self.R, input_terminal)
            #u_of_tx = self.net_dgm(t, input_terminal)
            u_of_tx = model(t, input_terminal)
            MSE_terminal = loss_fn(u_of_tx, exact_terminal)


            loss = 2 * MSE_functional + 0.2 * MSE_terminal
            loss.backward()
            optimizer.step()

            '''
            Plug in the testing data
            '''
            test_approximation = self.net_dgm(test_samples_t, test_samples_x)
            #loss_against_test_interior = loss_fn(test_exact_v_tx, test_approximation)
            loss_against_test_interior = torch.sum((abs(test_approximation-test_exact_v_tx) / test_exact_v_tx ) ** 2)
            losses_interior.append(loss_against_test_interior.item())

            #test_approximation_terminal = self.net_dgm(terminal_time_grid, test_samples_x_reshape)
            #loss_against_test_terminal = loss_fn(test_exact_v_tx_terminal_reshape, test_approximation_terminal)
            #losses_teminal.append(loss_against_test_terminal.item())

            if epoch%10 == 0:
                pbar.update(10)
                pbar.write("Iteration: {}/{}\t MSE interior: {:.4f}\t ".format(epoch, self.num_of_epochs, loss_against_test_interior.item()))#, loss_against_test_terminal.item()
        
        print(test_approximation, test_exact_v_tx)
        print(abs(test_approximation-test_exact_v_tx))
        return(losses_interior, model)#, losses_teminal)

class policy_iteration_training(nn.Module):
    def __init__(self, d, hidden_dim, num_samples, num_of_epochs, sigma, H, M, C, D, T, R, lr):
        super().__init__()
        self.dim = d 
        self.output_size_control = 2
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.num_of_epochs = num_of_epochs
        self.net_dgm = Net_DGM(dim_x = self.dim, dim_S = self.hidden_dim, activation='Tanh')
        self.FFN_dmg = FFN([self.dim + 1] + [self.hidden_dim, self.hidden_dim] + [self.output_size_control])
        self.sigma = sigma
        self.H = H
        self.M = M 
        self.C = C
        self.D = D
        self.T = T
        self.R = R
        self.lr = lr
        self.LQR_steps = 10000
        self.LQR_problem = SCDAA_with_changes.LQR(self.H, self.M, self.C, self.D, self.R, self.T, self.sigma, self.LQR_steps) 


    def train_initial_cost_functional_for_const_control(self, alpha):
        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss() 
        model = self.net_dgm
        #pbar = tqdm(total=self.num_of_epochs)
        for epoch in range(self.num_of_epochs):
            optimizer.zero_grad()

            train_samples_t, train_samples_x = generate_training_data(self.num_samples)
            t_batch, x_batch = train_samples_t.unsqueeze(1).float(), train_samples_x.float()
            t_batch.requires_grad = True
            x_batch.requires_grad = True
            
            train_u_tx = model(t_batch, x_batch)
            grad_u_x = get_gradient(train_u_tx,x_batch)
            grad_u_t = get_gradient(train_u_tx, t_batch)
            laplacian = get_laplacian(grad_u_x, x_batch)
            target_functional = torch.zeros_like(train_u_tx)
            grad_u_x_H_x = batch_Mat_batch(grad_u_x, self.H, x_batch)
            grad_u_x_M_alpha = batch_Mat_batch(grad_u_x, self.M, alpha)
            x_T_C_x = batch_Mat_batch(x_batch, self.C, x_batch)
            alpha_D_alpha = batch_Mat_batch(alpha, self.D, alpha)
            pde = grad_u_t + 0.5 * self.sigma ** 2 * laplacian + grad_u_x_H_x + grad_u_x_M_alpha + x_T_C_x + alpha_D_alpha
            MSE_functional = loss_fn(pde, target_functional)

            t = torch.ones(self.num_samples, 1) * self.T
            input_terminal = generate_training_data(self.num_samples)[1].float()
            exact_terminal = batch_Mat_batch(input_terminal, self.R, input_terminal)
            u_of_tx = model(t, input_terminal)
            MSE_terminal = loss_fn(u_of_tx, exact_terminal)

            loss = 2 * MSE_functional + 0.2 * MSE_terminal
            loss.backward()
            optimizer.step()
            
            #pbar.update(1)
        return(model)

    def learning_next_control(self, model_current_cost_functional):
        model = self.FFN_dmg
        # Define your loss function and optimizer
        def custom_loss_function(output, target): 
            return(torch.mean(output - target))
        def generate_training_data_reshape(num_samples):
            t_samples = np.random.uniform(0, 1, num_samples)
            x_samples = np.random.uniform(-3, 3, (num_samples, 2))
            return((torch.tensor(t_samples, dtype = torch.float64, requires_grad=True).float()).reshape(self.num_samples,1), 
                   torch.tensor(x_samples, dtype=torch.float64, requires_grad=True).float())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #pbar = tqdm(total=self.num_of_epochs)
        for epoch in range(self.num_of_epochs):
            t_train, x_train = generate_training_data_reshape(self.num_samples)
            inputs_train = torch.cat((t_train, x_train), dim = -1).float()

            # Forward pass
            optimizer.zero_grad()
            outputs_train = model(inputs_train)
            
            train_u_tx = model_current_cost_functional(t_train, x_train)
            grad_u_x = get_gradient(train_u_tx, x_train)
            grad_u_x_H_x = batch_Mat_batch(grad_u_x, self.H, x_train)
            grad_u_x_M_alpha = batch_Mat_batch(grad_u_x, self.M, outputs_train)
            x_T_C_x = batch_Mat_batch(x_train, self.C, x_train)
            alpha_D_alpha = batch_Mat_batch(outputs_train, self.D, outputs_train)
            output = grad_u_x_H_x + grad_u_x_M_alpha +x_T_C_x + alpha_D_alpha
            target = torch.zeros_like(train_u_tx)
            loss = custom_loss_function(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            #pbar.update(1)
        return(model)
    
    def learning_next_value_function(self, model_for_alpha):
        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss() 
        model = self.net_dgm
        #pbar = tqdm(total=self.num_of_epochs)
        for epoch in range(self.num_of_epochs):
            optimizer.zero_grad()

            train_samples_t, train_samples_x = generate_training_data(self.num_samples)
            t_batch, x_batch = train_samples_t.unsqueeze(1).float(), train_samples_x.float()
            t_batch.requires_grad = True
            x_batch.requires_grad = True
            alpha = model_for_alpha(torch.cat((t_batch.reshape(self.num_samples,1), x_batch), dim = -1))
            
            train_u_tx = model(t_batch, x_batch)
            grad_u_x = get_gradient(train_u_tx,x_batch)
            grad_u_t = get_gradient(train_u_tx, t_batch)
            laplacian = get_laplacian(grad_u_x, x_batch)
            target_functional = torch.zeros_like(train_u_tx)
            grad_u_x_H_x = batch_Mat_batch(grad_u_x, self.H, x_batch)
            grad_u_x_M_alpha = batch_Mat_batch(grad_u_x, self.M, alpha)
            x_T_C_x = batch_Mat_batch(x_batch, self.C, x_batch)
            alpha_D_alpha = batch_Mat_batch(alpha, self.D, alpha)
            pde = grad_u_t + 0.5 * self.sigma ** 2 * laplacian + grad_u_x_H_x + grad_u_x_M_alpha + x_T_C_x + alpha_D_alpha
            MSE_functional = loss_fn(pde, target_functional)

            t = torch.ones(self.num_samples, 1) * self.T
            input_terminal = generate_training_data(self.num_samples)[1].float()
            exact_terminal = batch_Mat_batch(input_terminal, self.R, input_terminal)
            u_of_tx = model(t, input_terminal)
            MSE_terminal = loss_fn(u_of_tx, exact_terminal)

            loss = 2 * MSE_functional + 0.2 * MSE_terminal
            loss.backward()
            optimizer.step()
            
            #pbar.update(1)
        return(model)





class linear_parabolic_PDE_solver_general_control(nn.Module):
    def __init__(self, d, hidden_dim, num_samples, num_of_epochs, sigma, H, M, C, D, T, R, lr, control):
        super().__init__()
        self.dim = d # dimension of input data 
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples # number of samples we want to use to train our model
        self.num_of_epochs = num_of_epochs

        self.net_dgm = Net_DGM(dim_x = self.dim, dim_S = self.hidden_dim, activation='Tanh')

        self.sigma = sigma
        self.H = H
        self.M = M 
        self.C = C
        self.D = D
        self.T = T
        self.R = R
        self.lr = lr
        self.control = control

        self.LQR_steps = 10000
        self.LQR_problem = SCDAA_with_changes.LQR(self.H, self.M, self.C, self.D, self.R, self.T, self.sigma, self.LQR_steps) 


    def train(self, test_samples_t, test_samples_x, test_exact_v_tx, alpha):
        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)
        loss_fn = nn.MSELoss() 

        pbar = tqdm(total=self.num_of_epochs)

        losses_interior = []
        #losses_teminal = []
        
        test_approximation = None
        #test_approximation_terminal = None
        #train_samples_t, train_samples_x = generate_training_data(self.num_samples)
        #t_batch, x_batch = train_samples_t.unsqueeze(1).float(), train_samples_x.float()
        #t_batch.requires_grad = True
        #x_batch.requires_grad = True

        model = self.net_dgm

        for epoch in range(self.num_of_epochs):
            optimizer.zero_grad()

            # generate the training data 
            #if epoch % self.gen_samples == 0:
            train_samples_t, train_samples_x = generate_training_data(self.num_samples)
            t_batch, x_batch = train_samples_t.unsqueeze(1).float(), train_samples_x.float()
            t_batch.requires_grad = True
            x_batch.requires_grad = True
            
            train_u_tx = self.net_dgm(t_batch, x_batch) # the current approximation of a cost functional with current 
            grad_u_x = get_gradient(train_u_tx,x_batch)
            grad_u_t = get_gradient(train_u_tx, t_batch)
            laplacian = get_laplacian(grad_u_x, x_batch)
            target_functional = torch.zeros_like(train_u_tx)
            grad_u_x_H_x = batch_Mat_batch(grad_u_x, self.H, x_batch)
            grad_u_x_M_alpha = batch_Mat_batch(grad_u_x, self.M, alpha)
            x_T_C_x = batch_Mat_batch(x_batch, self.C, x_batch)
            alpha_D_alpha = batch_Mat_batch(alpha, self.D, alpha)
            pde = grad_u_t + 0.5 * self.sigma ** 2 * laplacian + grad_u_x_H_x + grad_u_x_M_alpha + x_T_C_x + alpha_D_alpha
            MSE_functional = loss_fn(pde, target_functional)

            '''
            Now do the terminal condition 
            '''
            t = torch.ones(self.num_samples, 1) * self.T
            input_terminal = generate_training_data(self.num_samples)[1].float()
            exact_terminal = batch_Mat_batch(input_terminal, self.R, input_terminal)
            u_of_tx = self.net_dgm(t, input_terminal)
            MSE_terminal = loss_fn(u_of_tx, exact_terminal)


            loss = 2 * MSE_functional + 0.2 * MSE_terminal
            loss.backward()
            optimizer.step()

            '''
            Plug in the testing data
            '''
            test_approximation = self.net_dgm(test_samples_t, test_samples_x)
            #loss_against_test_interior = loss_fn(test_exact_v_tx, test_approximation)
            loss_against_test_interior = torch.sum((abs(test_approximation-test_exact_v_tx) / test_exact_v_tx ) ** 2)
            losses_interior.append(loss_against_test_interior.item())

            #test_approximation_terminal = self.net_dgm(terminal_time_grid, test_samples_x_reshape)
            #loss_against_test_terminal = loss_fn(test_exact_v_tx_terminal_reshape, test_approximation_terminal)
            #losses_teminal.append(loss_against_test_terminal.item())

            if epoch%10 == 0:
                pbar.update(10)
                pbar.write("Iteration: {}/{}\t MSE interior: {:.4f}\t ".format(epoch, self.num_of_epochs, loss_against_test_interior.item()))#, loss_against_test_terminal.item()
        
        print(test_approximation, test_exact_v_tx)
        print(abs(test_approximation-test_exact_v_tx))
        return(losses_interior, self.net_dgm)#, losses_teminal)
        


