import numpy as np
import torch
from scipy.integrate import odeint
import itertools
import matplotlib.pyplot as plt
import torch.nn as nn
from torchdiffeq import odeint as odeint_2

class LQR:
    def __init__(self, H, M, C, D, R, T, sigma):
        self.H = np.array(H)
        self.M = np.array(M)
        self.C = np.array(C)
        self.D = np.array(D)
        self.D_inv = np.linalg.inv(D)
        self.R = np.array(R)
        self.T = T
        self.sigma = np.array(sigma)
        self.sigsig_T = torch.matmul(sigma, sigma.T) 
    
    def solve_ode(self, steps):
        MDM = self.M @ np.linalg.inv(self.D) @ self.M.T
        def odefunc(S_vec, t):
            S = np.reshape(S_vec, (2, 2))
            S_prime = -2 * np.dot(self.H.T, S) + S @ MDM @ S - self.C
            S_prime_vec = S_prime.flatten()
            return S_prime_vec
        S0 = self.R.flatten().copy()
        t_vals = np.linspace(self.T, 0, steps)
        sol = odeint(odefunc, S0, t_vals, atol=1e-11, rtol=1e-11)
        solution_matrix = [ np.reshape(sol[i, :], (2,2)) for i in range(len(sol))]
        solution_matrix = np.flip(solution_matrix,0)
        solution_matrix = solution_matrix.copy()
        return(torch.from_numpy(solution_matrix))
    
    def TrSigSig_T(self, r_sol):
        output = torch.matmul(self.sigsig_T, r_sol)
        return(output.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1))

    def construct_value_funct(self, space_grid, r_sol):
        step_size = self.T / ( len(r_sol) - 1 )
        sig_sig_Tr = self.TrSigSig_T(r_sol)
        interp = ( sig_sig_Tr[0:-1] + sig_sig_Tr[1:] ) / 2
        value_funct = torch.zeros(len(r_sol), len(space_grid))
        for i in range(len(r_sol)):
            temp = torch.sum(r_sol[i] * space_grid.unsqueeze(2), axis = 1)
            value_funct[i,:] = torch.sum(temp * space_grid, axis = 1) 
        integral = torch.cumsum(interp.flip(0), 0).flip(0) * step_size
        for i in range(len(r_sol) - 1): 
            value_funct[i,:] = value_funct[i,:] + integral[i]
        return(value_funct)
    
    def generate_trajectories(self, x, MC_, N, r_sol):
        # x = initial_condition
        # M = number of MC samples
        # N = time_steps + 1
        M_ten = torch.tensor(self.M, dtype = torch.float64)
        H_ten = torch.tensor(self.H, dtype = torch.float64)
        D_inv_ten = torch.tensor(self.D_inv, dtype = torch.float64)
        sigma_ten_T = torch.tensor(self.sigma.T, dtype=torch.float64)

        del_T = self.T / (N - 1)
        sqrt_dt = np.sqrt(del_T)
        id_ = torch.tensor([[1,0],[0,1]], dtype = torch.float64)
        trajectories = torch.zeros(MC_,N,2, dtype = torch.float64)
        trajectories[:,0,:] = x
        #multiplier = torch.transpose(id_ + del_T * (self.H - self.M @ self.D_inv @ self.M.T @ r_sol), 1,2)
        multiplier = torch.transpose(id_ + del_T * (H_ten - M_ten @ D_inv_ten @ M_ten.T @ r_sol), 1,2)
        for i in range(1, len(r_sol)):
            trajectories[:,i,:] = torch.matmul(trajectories[:,i-1,:],multiplier[i-1]) + sqrt_dt * torch.matmul(torch.randn(MC_, 2, dtype = torch.float64) , sigma_ten_T)
        return(torch.sum(trajectories,axis = 0) / MC_)
    
    def generate_number(self, space_grid,MC_):
        return(torch.randn(len(space_grid), MC_, 2, dtype = torch.float64))
    
    def generate_trajectories_grid(self, space_grid, MC_, N, r_sol):
        # space_grid = the space grid
        # M = number of MC samples
        # N = time_steps + 1

        M_ten = torch.tensor(self.M, dtype = torch.float64)
        H_ten = torch.tensor(self.H, dtype = torch.float64)
        D_inv_ten = torch.tensor(self.D_inv, dtype = torch.float64)
        sigma_ten_T = torch.tensor(self.sigma.T, dtype=torch.float64)

        del_T = self.T / (N - 1)
        sqrt_dt = np.sqrt(del_T)
        id_ = torch.tensor([[1,0],[0,1]], dtype = torch.float64)

        average_trajectories = torch.zeros(len(space_grid), len(r_sol), 2)
        average_trajectories[:,0,:] = space_grid

        trajectories_old = torch.zeros(len(space_grid), MC_, 2, dtype = torch.float64)
        trajectories_old = space_grid.unsqueeze(1).repeat(1, MC_, 1)
        
        multiplier = torch.transpose(id_ + del_T * (H_ten - M_ten @ D_inv_ten @ M_ten.T @ r_sol), 1,2)

        for i in range(1, len(r_sol)):
            random = torch.randn(len(space_grid), MC_, 2, dtype = torch.float64)
            trajectories_old = torch.matmul(trajectories_old, multiplier[i-1]) + sqrt_dt * torch.matmul( random, sigma_ten_T)
            average_trajectories[:,i,:] = torch.sum(trajectories_old, axis = 1) / MC_
        return(average_trajectories)

    
    def MC_value_function_x(self, x, r_sol, MC_):
        M_ten = torch.tensor(self.M, dtype = torch.float64)
        D_inv_ten = torch.tensor(self.D_inv, dtype = torch.float64)
        MC_value_funct = torch.zeros(len(r_sol))
        R_ten = torch.tensor(self.R, dtype = torch.float64)
        trajectory = self.generate_trajectories(x, MC_, len(r_sol), r_sol)

        optimal_a_x = torch.matmul(D_inv_ten @ M_ten.T @ r_sol,trajectory.unsqueeze(-1)).squeeze(-1) 
        
        F_trajectory = torch.sum(torch.matmul(trajectory, C) * trajectory, axis = 1) + torch.sum(torch.matmul(optimal_a_x, D) * optimal_a_x, axis = 1)
        F_interp = ( F_trajectory[0:-1] + F_trajectory[1:] ) * (1 / 2) * ( 1 / (len(r_sol) - 1) ) 

        F_integrated = torch.flip(torch.cumsum(F_interp, axis =  0), dims = (0,))
        MC_value_funct[0:-1] = F_integrated
        terminal_times = torch.flip(torch.sum(torch.matmul(trajectory, R_ten) * trajectory, axis = 1), dims=(0,))
        MC_value_funct = MC_value_funct + terminal_times
        return(MC_value_funct)

    def approx_value_function(self, space_grid, r_sol, ave_traj):
        MC_approx = torch.zeros(len(space_grid), len(r_sol))
        M_ten = torch.tensor(self.M, dtype = torch.float64)
        D_inv_ten = torch.tensor(self.D_inv, dtype = torch.float64)
        MC_value_funct = torch.zeros(len(r_sol))
        R_ten = torch.tensor(self.R, dtype = torch.float64)


        return(MC_approx)

if __name__ == '__main__':
    T = 1
    H = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)*0.1
    M = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
    sigma = torch.tensor([[0, 0], [0, 0]], dtype = torch.float64)
    C = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
    D = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
    R = torch.tensor([[1, 0], [0,1]], dtype = torch.float64) * 10
    spatial_grid_1D = torch.linspace(0, 1, 2)
    spatial_grid = torch.tensor(list(itertools.product(spatial_grid_1D, repeat=2)), dtype = torch.float64)

    errors = []
    r_errors = []
    list_time_steps = [9,5,50, 100, 500, 1000, 5000]#, 10000, 50000]
    
    MC = 15000
    
    time_grid_ric = torch.linspace(0, T, 10000)
    LQR_problem_ric = LQR(H, M, C, D, R, T, sigma)
    sol_ric = LQR_problem_ric.solve_ode(len(time_grid_ric))
    value_funct_ric = LQR_problem_ric.construct_value_funct(spatial_grid, sol_ric)

    for i in range(len(list_time_steps)):
        time_steps = list_time_steps[i]
        time_grid = torch.linspace(0, T, time_steps + 1)

        LQR_problem = LQR(H, M, C, D, R, T, sigma)
        sol = LQR_problem.solve_ode(len(time_grid))
        #value_funct = LQR_problem.construct_value_funct(spatial_grid, sol)
        
        #traj = LQR_problem.generate_trajectories(spatial_grid[2], MC, time_steps+1, sol)
        #ave_traj_grid = LQR_problem.generate_trajectories_grid(spatial_grid, MC, time_steps+1, sol)
        
        #MC_value_funct = LQR_problem.approx_value_function(spatial_grid, sol, ave_traj_grid)

        MC_value_funct = LQR_problem.MC_value_function_x(spatial_grid[-1], sol, MC)
        
        L_1_Error = torch.sum(abs(value_funct[0,-1] - MC_value_funct[0])).item()
        errors.append(L_1_Error)

    xpoints=list_time_steps
    ypoints=errors
    plt.loglog(xpoints,ypoints,marker = 'o')
    plt.title('With noise, M = 3000- varying time steps, look at error at t = 0, x = [1,1]')
    plt.xlabel("Number of time steps")
    plt.ylabel("Error")
    plt.show()
        
    print(errors)





