import numpy as np
import torch
#torch.manual_seed(0)
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import itertools
import matplotlib.pyplot as plt

class LQR:
    def __init__(self, H, M, C, D, R, T, sigma,steps):
        self.H = np.array(H)
        self.M = np.array(M)
        self.M_T = np.transpose(self.M)
        self.M_T_Trans = torch.tensor(self.M_T, dtype = torch.float64)
        self.C = np.array(C)
        self.D = np.array(D)
        self.D_ten = torch.tensor(self.D, dtype = torch.float64)
        self.D_inv = np.linalg.inv(D)
        self.D_inv_Ten = torch.tensor(self.D_inv, dtype = torch.float64)
        self.R = np.array(R)
        self.R_ten = torch.tensor(self.R , dtype = torch.float64)
        self.T = T
        self.sigma = np.array(sigma)
        self.steps = steps
        self.t_vals = torch.linspace(0,self.T, steps)
        self.dt = self.t_vals[1]-self.t_vals[0]
        self.dW = np.sqrt(self.dt)
        self.H_ten = torch.tensor(self.H, dtype = torch.float64)
        self.D_inv_M_T = torch.tensor(self.D * self.M_T, dtype = torch.float64)
        self.C_ten = C

    def solve_ode(self, steps):
        MDM = self.M @ np.linalg.inv(self.D) @ self.M.T
        def odefunc(S_vec, t):
            S = np.reshape(S_vec, (2, 2))
            S_prime = -2 * np.dot(self.H.T, S) + S @ MDM @ S - self.C
            S_prime_vec = S_prime.flatten()
            return S_prime_vec
        S0 = self.R.flatten().copy()
        t_vals = np.linspace(self.T, 0, steps)
        sol = odeint(odefunc, S0, t_vals, atol=1e-15, rtol=1e-13)
        solution_matrix = [ np.reshape(sol[i, :], (2,2)) for i in range(len(sol))]
        solution_matrix = np.flip(solution_matrix,0)
        solution_matrix = solution_matrix.copy()
        return(torch.from_numpy(solution_matrix))
    
    def solve_ode_2(s):
        S0 = s.R.reshape(2*2)
        t_min = s.t_vals[0]
        t_max = s.t_vals[-1]
        def ode_rhs(t, S, M, H, C, D):
            # Define the Ricatti ODE
            D_inv = np.linalg.inv(D)
            S = S.reshape((2,2))
            
            M_T_x_S = np.matmul(M.T,S)
            D_inv_x_M_T_S = np.matmul(D_inv, M_T_x_S)
            SxM = np.matmul(S,M)
            
            dS_dt = 2.0*np.matmul(H.T,S) - np.matmul(SxM,D_inv_x_M_T_S)  + C 
            dS_dt = dS_dt.reshape(2*2)
            return dS_dt
        sol = solve_ivp(ode_rhs, (t_min, t_max), S0, args=(s.M, s.H, s.C, s.D), t_eval=s.t_vals, atol=1e-15, rtol=1e-13)
        output = np.zeros((s.steps,2,2))
        for i in range(s.steps): 
            output[i] = sol.y[:,s.steps-i-1].reshape(2,2)
        return(torch.from_numpy(output))
    
    def closest_point_in_grid(self, t, N):
        delta_t = (self.t_vals[1] - self.t_vals[0]).item()  # Length of each interval
        index = round(t / delta_t)  # Find the index of the closest interval
        index = min(max(index, 0), N - 1)  # Ensure index is within bounds
        return(index)

    def construct_value_funct(self, t, x, r_sol):
        '''
        t = is a number in [0,1], we first need to an appropriate index
        on the time grid r_sol is defined on.  
        For a fixed t and x we return v(t,x)
        '''
        t_index = self.closest_point_in_grid(t, len(r_sol))
        step_size = self.t_vals[1] - self.t_vals[0] 
        step_sizes = self.t_vals[t_index + 1:] - self.t_vals[t_index:-1]
        integral = 0
        output = r_sol[t_index:] * self.sigma * self.sigma
        Tr_output = output.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        integral = torch.sum(Tr_output[:-1]) * step_size
        #integral = torch.sum(Tr_output[:-1]  * step_sizes)
        xS_t = torch.matmul(x,r_sol[t_index])
        xS_tx = torch.dot(xS_t,x)
        return(integral + xS_tx)

    def construct_value_funct_batch(self, t_batch, x_batch, r_sol):
        #t_batch = torch.tensor(t_batch_arr.clone().detach(), dtype = torch.float64)
        #x_batch = torch.tensor(x_batch_arr.clone().detach(), dtype = torch.float64)
        #t_batch.to(torch.float64)
        x_batch = x_batch.to(torch.float64)

        val_funct = torch.zeros(len(x_batch))
        # first need the indices 
        t_grid_batch = self.t_vals.expand(len(t_batch), -1 )
        starting_indices = torch.argmin(np.abs(t_grid_batch - t_batch.reshape(len(t_batch), 1)), dim = 1)
        reversed_indices = self.steps - starting_indices
        step_size = self.t_vals[1] - self.t_vals[0]

        sigma_sigma_r_sol_flip = torch.flip(self.sigma * self.sigma * r_sol, dims = (0,))
        Tr_sigma_sigma_r_sol_flip = sigma_sigma_r_sol_flip.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) * step_size
        int_Tr = torch.cumsum(Tr_sigma_sigma_r_sol_flip[1:], dim = 0)

        for i in range(len(t_batch)):
            xS_t = torch.matmul(x_batch[i],r_sol[starting_indices[i]])
            xS_tx = torch.dot(xS_t,x_batch[i])
            val_funct[i] = int_Tr[reversed_indices[i]-2] + xS_tx
        return(val_funct)

    def construct_optimal_control(self, t, x, r_sol):
        t_index = self.closest_point_in_grid(t, len(r_sol))
        output = self.D_inv_M_T  * r_sol[t_index]
        return(-torch.matmul(output, x))

    def construct_optimal_control_batch(self, t_batch, x_batch, r_sol):
        optimal_control_batch = torch.zeros(len(t_batch), 2)
        t_grid_batch = self.t_vals.expand(len(t_batch), -1 )
        starting_indices = torch.argmin(np.abs(t_grid_batch - t_batch.reshape(len(t_batch), 1)), dim = 1)
        for i in range(len(t_batch)):
            optimal_control_batch[i] = torch.matmul(self.D_inv_M_T  * r_sol[starting_indices[i]], x_batch[i])
        return(-optimal_control_batch)

    def F(self, x_batch, S_i):
        '''
        x_batch = i_th step in all the M trajectories
        F(x_batch) = x^T C x + a(x)^TDa(x)
        '''
        x_TCx = torch.sum(torch.matmul(x_batch, C) * x_batch, axis = 1)
        a = - self.D_inv_Ten @ self.M_T_Trans @ S_i
        a_x = torch.matmul(x_batch, a.T)
        a_xDa_x = torch.sum(torch.matmul(a_x, C) * a_x, axis = 1)
        return(x_TCx + a_xDa_x)
    

    

    def step_in_x(self, i, X_0, M_d_inv_M_trans_S_i, bH,M_):
        # bH = torch.unsqueeze(self.H, 0).repeat(numMC,1,1)

        dt = self.t_vals[i+1]-self.t_vals[i]
        
        M_D_inv_M_trans_S_t = M_d_inv_M_trans_S_i
        b_M_D_inv_M_trans_S_t = torch.unsqueeze(M_D_inv_M_trans_S_t,0).repeat(M_,1,1)
        
        dW = np.sqrt(dt)*torch.randn(M_)
        X0_unsq = X_0.unsqueeze(2)
        result_from_x = torch.bmm(bH,X0_unsq).squeeze(2)
        result_from_a =  - torch.bmm(b_M_D_inv_M_trans_S_t,X0_unsq).squeeze(2)
        X1 = X_0 + result_from_x * dt + result_from_a * dt + self.sigma.item()*dW.unsqueeze(1)
        return(X1)

    def MC_approximation(self, t, x, r_sol, M_):
        t_index = self.closest_point_in_grid(t, len(r_sol))
        integral = 0
        MD_invM_T= torch.tensor(self.M @ self.D_inv @ self.M_T, dtype = torch.float64)
        MD_invM_TS = torch.matmul(MD_invM_T, r_sol[t_index:])
        H_MD_invM_TS = self.H_ten - MD_invM_TS
        x_0 = x.repeat(M_,1)

        H_ten = torch.tensor(self.H, dtype = torch.float64)
        bH= torch.unsqueeze(H_ten, 0).repeat(M_,1,1)

        for i in range(len(r_sol) - t_index - 1):
            #noise = self.dW * self.sigma*torch.randn(M_).unsqueeze(1) 
            dt = (self.t_vals[t_index + i + 1] - self.t_vals[t_index + i]).item()
            dW = np.sqrt(dt).item()
            #noise = self.dW * self.sigma*torch.randn(M_,2)
            noise = dW * self.sigma.item()*torch.randn(M_,2)
            #noise = dW* self.sigma*torch.randn(M_).unsqueeze(1)
            #x_1 = x_0 + self.dt * torch.matmul(x_0, H_MD_invM_TS[i]) + noise
            x_1 = x_0 + dt * torch.matmul(x_0, H_MD_invM_TS[i]) + noise

            #x_1 = self.step_in_x(t_index + i, x_0, MD_invM_TS[i], bH, M_)
            #integral = integral + self.F(x_1, r_sol[t_index + i]) * self.dt 
            integral = integral + self.F(x_1, r_sol[t_index + i]) * dt 
            x_0 = x_1
        terminal_condition = torch.sum(torch.matmul(x_0, R) * x_0, axis = 1)

        mean = torch.sum(integral + terminal_condition, axis = 0) / M_
        std = torch.std(integral + terminal_condition)/ np.sqrt(M_)
        return(mean.item(),std.item())
    
    
    def F_constant_control(self, x_batch, t, a_batch):
        '''
        x_batch = i_th step in all the M trajectories
        t = value of t
        '''
        x_TCx = torch.sum(torch.matmul(x_batch, self.C_ten) * x_batch, axis = 1)
        aDa = torch.sum( torch.matmul(a_batch, self.D_ten) * a_batch, axis = 1 )
        return(x_TCx + aDa)
    
    def step_in_x_constant_control(self, i, X_0,  bH, M_, b_a, b_M):
        # bH = torch.unsqueeze(self.H, 0).repeat(numMC,1,1)

        dt = self.t_vals[i+1]-self.t_vals[i]
        dW = np.sqrt(dt)*torch.randn(M_)
        X0_unsq = X_0.unsqueeze(2)
        b_a_unsq = b_a.unsqueeze(2)
        result_from_x = torch.bmm(bH,X0_unsq).squeeze(2)
        result_from_a = torch.bmm(b_M, b_a_unsq).squeeze(2)
        X1 = X_0 + (result_from_x+result_from_a) * dt + + self.sigma.item() * dW.unsqueeze(1)
        return(X1)

    def MC_approximation_constant_control(self, t, x, M_):
        t_index = self.closest_point_in_grid(t, self.steps)
        integral = torch.zeros(1)
        x_0 = x.repeat(M_,1) 
        x_0_new = x.repeat(M_,1) 

        H_ten = torch.tensor(self.H, dtype = torch.float64)
        bH= torch.unsqueeze(H_ten, 0).repeat(M_,1,1)

        M_ten = torch.tensor(self.M_T, dtype = torch.float64)
        bM= torch.unsqueeze(M_ten, 0).repeat(M_,1,1)
        
        a_batch = torch.tensor([1,1], dtype = torch.float64).repeat(M_,1) 
        Ma = torch.matmul(a_batch.unsqueeze(1), bM).squeeze(1)

        for i in range(self.steps - t_index - 1):
            dt = (self.t_vals[t_index + i + 1] - self.t_vals[t_index + i]).item()
            #dW = np.sqrt(dt).item()
            #noise = dW * self.sigma.item()*torch.randn(M_,2)
            #x_1 = x_0 + dt * ( torch.matmul(x_0, H_ten) + Ma) + noise
            x_1 = self.step_in_x_constant_control(i, x_0,  bH, M_, a_batch, bM)
            integral = integral + self.F_constant_control(x_1,t,a_batch) * dt 
            x_0 = x_1

        terminal_condition = torch.sum(torch.matmul(x_0, self.R_ten) * x_0, axis = 1)
        integral_mean = torch.sum(integral, axis = 0) / M_
        terminal_mean = torch.sum(terminal_condition, axis = 0) / M_

        mean = torch.sum(integral + terminal_condition, axis = 0) / M_
        return(mean.item())

# check how long to run 10000 samples 

if __name__ == '__main__':
    torch.set_printoptions(precision=10)
    print('Hello Worl')
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
    LQR_problem_ric = LQR(H, M, C, D, R, T, sigma, LQR_steps)
    #sol = LQR_problem_ric.solve_ode(LQR_steps)
    sol = LQR_problem_ric.solve_ode_2()
    X0 = torch.tensor([1,1], dtype = torch.float64)
    value_funct = LQR_problem_ric.construct_value_funct(t = 0, x = X0, r_sol = sol)
    optimal_control = LQR_problem_ric.construct_optimal_control(t = 0, x = X0, r_sol = sol)
    
    
    '''
    batch_size = 10
    t_samples = np.random.uniform(0, 1, batch_size)
    x_samples = np.random.uniform(-3, 3, (batch_size, 2))
    t_batch_ten = torch.tensor(t_samples, dtype = torch.float64)
    x_batch_ten = torch.tensor(x_samples, dtype = torch.float64)
    
    v_funct = LQR_problem_ric.construct_value_funct_batch(t_batch_ten,x_batch_ten, sol)
    print(v_funct)
    error = 0
    for i in range(len(t_samples)):
        approx = LQR_problem_ric.construct_value_funct(t = t_batch_ten[i].item(), x = x_batch_ten[i], r_sol = sol)
        print(t_samples[i], x_samples[i], approx)
        error = error + np.abs(approx.item() - v_funct[i].item())
    print(error)
    '''
    
    
    MC = 500000
    time_steps_array = [10-1,50-1,100-1,500-1,1000-1,5000-1,10_000-1]
    v_lb_mc_array = np.zeros(len(time_steps_array))
    v_ub_mc_array = np.zeros(len(time_steps_array))
    v_mc_array = np.zeros(len(time_steps_array))
    v_ricatti_array = np.ones(len(time_steps_array)) * value_funct.item()

    error_list = np.zeros(len(time_steps_array))
    for i in range(len(time_steps_array)):
        n_time_steps = time_steps_array[i]
        LQR_MC = LQR(H, M, C, D, R, T, sigma, n_time_steps + 1)
        time_grid = torch.linspace(0, T, n_time_steps + 1)
        #sol = LQR_MC.solve_ode(n_time_steps + 1)
        sol = LQR_MC.solve_ode_2()
        MC_value_funct = LQR_MC.MC_approximation(t = 0, x = X0, r_sol = sol, M_ = MC)
        error_list[i] = abs(MC_value_funct[0] - value_funct)

        conf_bound = 1.96*MC_value_funct[1]
        v_mc_array[i] = MC_value_funct[0]
        v_lb_mc_array[i] = MC_value_funct[0] - conf_bound
        v_ub_mc_array[i] = MC_value_funct[0] + conf_bound

        print(error_list)

    
    plt.plot(np.log(time_steps_array), 1.5 - 1.0 * np.log(time_steps_array),color='red',label='theoretical rate 1')
    plt.plot(np.log(time_steps_array), np.log(error_list),marker = 'o')
    plt.xlabel('Number of time Steps varying')
    plt.ylabel('Absolute Error of v(t,x)')
    plt.title('Log-Log plot of absolute error with MC samples = ' + str(MC))
    plt.savefig("lqr_convergence_time_steps.pdf")
    plt.close()




    plt.plot(time_steps_array, v_mc_array, color='black',label='MC estimate')
    plt.plot(time_steps_array, v_ricatti_array, color='red',label='True val from Ricatti')
    plt.plot(time_steps_array, v_ub_mc_array, color='green',label='MC estimate UB at 95%')
    plt.plot(time_steps_array, v_lb_mc_array, color='green',label='MC estimate LB at 95%')
    plt.xlabel('number of time steps')
    plt.ylabel('v(t,x) and confidence bounds')
    plt.title('MC and Ricatti values for v(t,x) and confidence bounds')
    plt.legend()
    plt.savefig("lqr_v_values_time_steps.pdf")
    plt.close()



    






    