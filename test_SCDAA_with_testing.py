import unittest
import SCDAA_with_testing
import SCDAA_with_changes
import torch
import itertools
import numpy as np


class TestSCDAA(unittest.TestCase):
    
    def test_ricatti_sol_1(self):
        # create and instance of an LQR problem
        T = 1
        H = torch.tensor([[1,0], [0,0]], dtype = torch.float64)
        M = torch.tensor([[1,0], [0,0]], dtype = torch.float64)
        sigma = torch.tensor([[0,0], [0,0]], dtype = torch.float64)
        C = torch.tensor([[0,0], [0,0]], dtype = torch.float64)
        D = torch.tensor([[1,0], [0,1]], dtype = torch.float64)
        R = torch.tensor([[1,0], [0,1]], dtype = torch.float64)
        test_LQR = SCDAA_with_testing.LQR(H, M, C, D, R, T, sigma)
        time_steps = 10000
        test_time_grid = torch.linspace(0, T, time_steps + 1)
        ricatti_sol =  test_LQR.solve_ode(time_steps + 1)
        exact_sol = 2 / ( 1 + torch.exp( 2 * (test_time_grid - 1) ) )
        truth = torch.round(torch.sum(abs(ricatti_sol[:,0,0]-exact_sol), axis = 0), decimals = 4)
        self.assertLess(truth.item(), 10**-3)

    def test_ricatti_sol_2(self):
        # create and instance of an LQR problem
        T = 1
        H = torch.tensor([[0,0], [0,1]], dtype = torch.float64)
        M = torch.tensor([[0,0], [0,1]], dtype = torch.float64)
        sigma = torch.tensor([[0,0], [0,0]], dtype = torch.float64)
        C = torch.tensor([[0,0], [0,0]], dtype = torch.float64)
        D = torch.tensor([[1,0], [0,1]], dtype = torch.float64)
        R = torch.tensor([[1,0], [0,1]], dtype = torch.float64)
        test_LQR = SCDAA_with_testing.LQR(H, M, C, D, R, T, sigma)
        time_steps = 10000
        test_time_grid = torch.linspace(0, T, time_steps + 1)
        ricatti_sol =  test_LQR.solve_ode(time_steps + 1)
        exact_sol = 2 / ( 1 + torch.exp( 2 * (test_time_grid - 1) ) )
        truth = torch.round(torch.sum(abs(ricatti_sol[:,1,-1]-exact_sol), axis = 0), decimals = 4)
        self.assertLess(truth.item(), 10**-3)
    
    def test_value_funct(self):
        # create and instance of an LQR problem
        T = 1
        H = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
        M = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
        sigma = torch.tensor([[1, 2], [1, 1]], dtype = torch.float64)
        C = torch.tensor([[1, 0], [1, 0]], dtype = torch.float64)
        D = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
        R = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
        test_LQR = SCDAA_with_testing.LQR(H, M, C, D, R, T, sigma)
        time_steps = 3
        ricatti_sol =  test_LQR.solve_ode(time_steps + 1)
        spatial_grid_1D = torch.linspace(0, 1, 2, dtype = torch.float64)
        spatial_grid = torch.tensor(list(itertools.product(spatial_grid_1D, repeat=2)))
        value_funct = test_LQR.construct_value_funct(spatial_grid, ricatti_sol)
        value_to_check = -1
        x = spatial_grid[value_to_check]
        temp = torch.matmul(torch.matmul(x, ricatti_sol[0]),x)
        helper = test_LQR.TrSigSig_T(ricatti_sol)
        helper2 = (T / time_steps) *  torch.sum( ( helper[0:-1] + helper[1:]) / 2 , axis = 0)
        exact_value = temp + helper2
        self.assertLess(abs(exact_value.item()-value_funct[0,value_to_check].item()), 10**-4)
    
    

    def test_optimised_trajectory(self):
        T = 1
        H = torch.tensor([[1, 0], [0, 0]], dtype = torch.float64)
        M = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
        sigma = torch.tensor([[0, 0], [0, 0]], dtype = torch.float64)
        C = torch.tensor([[1, 0], [1, 0]], dtype = torch.float64)
        D = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
        R = torch.tensor([[1, 0], [0, 1]], dtype = torch.float64)
        time_steps = 200
        LQR_problem_test = SCDAA_with_testing.LQR(H, M, C, D, R, T, sigma)
        sol = LQR_problem_test.solve_ode(time_steps + 1)
        spatial_grid_1D = torch.linspace(0, 1, 3)
        spatial_grid = torch.tensor(list(itertools.product(spatial_grid_1D, repeat=2)), dtype = torch.float64)
        MC = 1000
        traj_grid = LQR_problem_test.generate_trajectories_grid(spatial_grid, MC, time_steps+1, sol)
        output = 0
        for i in range(1,len(spatial_grid)): 
            a = LQR_problem_test.generate_trajectories(spatial_grid[i], MC, time_steps+1, sol)
            b = traj_grid[i] 
            output = output + torch.sum(abs(a - b)).item()
        self.assertAlmostEqual(output / torch.sum(abs(traj_grid)).item(), 0, 3)
    
    def test_batch_value_funct(self):

        batch_size = 10
        t_samples = np.random.uniform(0, 1, batch_size)
        x_samples = np.random.uniform(-3, 3, (batch_size, 2))
        v_funct = LQR_problem_ric.construct_value_funct_batch(t_samples,x_samples)
        
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
        #sol = LQR_problem_ric.solve_ode(LQR_steps)
        sol = LQR_problem_ric.solve_ode_2()
        X0 = torch.tensor([1,1], dtype = torch.float64)
        value_funct = LQR_problem_ric.construct_value_funct(t = 0, x = X0, r_sol = sol)

        batch_size = 10
        t_samples = np.random.uniform(0, 1, batch_size)
        x_samples = np.random.uniform(-3, 3, (batch_size, 2))
        v_funct = LQR_problem_ric.construct_value_funct_batch(t_samples,x_samples, sol)
        print(v_funct)
        error = 0
        for i in range(len(t_samples)):
            approx = LQR_problem_ric.construct_optimal_control(t = t_samples[i], x = x_samples[i], r_sol = sol)
            print(t_samples[i], x_samples[i], approx)
            error = error + np.abs(approx.item() - v_funct[i].item())
        print(error)
if __name__ == '__main__':
    unittest.main()