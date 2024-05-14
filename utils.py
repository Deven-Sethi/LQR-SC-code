import torch
import torch.nn as nn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

class LInfinityLoss(torch.nn.Module):
    def __init__(self):
        super(LInfinityLoss, self).__init__()
        
    def forward(self, prediction, target):
        loss = torch.max(torch.abs(prediction - target))
        return loss

class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
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

    def forward(self, t, x):
        tx = torch.cat((t,x), dim=1)
        return self.net(tx)

class Net_Action(nn.Module):
    def __init__(self):
        inputDim = 3
        hiddenWidth = 100
        outputDim = 2
    
        super(Net_Action, self).__init__()
        self.fc1 = nn.Linear(inputDim, hiddenWidth)
        self.fc2 = nn.Linear(hiddenWidth,hiddenWidth)
        # self.fc3 = nn.Linear(hiddenWidth,hiddenWidth)
        self.fc_out = nn.Linear(hiddenWidth, outputDim)


    def forward(self, t, x):
        tx = torch.cat((t,x), dim=1)
        tx = self.fc1(tx)
        tx = torch.relu(tx)
        
        tx = self.fc2(tx)
        tx = torch.relu(tx)
        
        # tx = self.fc3(tx)
        # tx = torch.relu(tx)
        
        tx = self.fc_out(tx)
        return tx



class DGM_Layer(nn.Module):
    
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


def generateTestValsX(batch_size, dimX, device):
    return torch.randn(batch_size, dimX, device=device) * 3

def generateTestValsT(batch_size, T, device):
    return torch.rand(batch_size, 1, device=device) * T    

def getGradient(u_of_tx, tx):
    d = torch.autograd.grad(u_of_tx, tx, grad_outputs = torch.ones_like(u_of_tx), create_graph=True, retain_graph=True, only_inputs=True)[0]    
    return d

def getLaplacian(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess_diag.append(grad2[:,d].view(-1,1))    
    hess_diag = torch.cat(hess_diag,1)
    laplacian = hess_diag.sum(1, keepdim=True)
    return laplacian

def getBatchVecTransMatrixVec(v,M):
    batch_size = v.size(0)
    v_size = v.size(1)
    
    # M must be square and same dimension as v
    assert(M.size(0)==M.size(1))
    assert(M.size(0)==v_size)
    
    # make copies so we don't modify inputs
    v_bar = v.reshape(batch_size,v_size,1)
    M_bar = M.repeat(batch_size,1,1)

    Mv = torch.bmm(M_bar,v_bar)
    v_transpose = v_bar.transpose(1,2)
    vMv = torch.bmm(v_transpose,Mv)
    return vMv

def getBatchVec1MatrixVec2(v1,M,v2):
    
    v1_size = v1.size(1)
    v2_size = v2.size(1)
    # M must be square and same dimension as v
    assert(M.size(0)==M.size(1))
    assert(M.size(0)==v1_size)
    assert(M.size(0)==v2_size)
    
    batch_size = v1.size(0)
    M_bar = M.repeat(batch_size,1,1)
    v2_bar = v2.reshape(batch_size,v2_size,1)
    Mv2 = torch.bmm(M_bar,v2_bar)

    v1_bar = v1.reshape(batch_size,v1_size,1)
    v1_transpose = v1_bar.transpose(1,2)
    v1Mv2 = torch.bmm(v1_transpose,Mv2)
    return v1Mv2


def plotTrained(u, T, dimX, dim=0):
    x_np = np.linspace(-6,6,100)
    t = torch.ones(1,np.size(x_np)).t() * T
    x0 = torch.reshape(torch.from_numpy(x_np), (np.size(x_np),1)).float()
    x = x0
    if dimX > 1:
        x1 = torch.zeros(np.size(x_np),dimX-1)
        x = torch.cat((x0,x1),dim=1)
    
    out = u(t,x)
    #print(out.shape)
    u_net = out.detach().numpy()
    plt.plot(x_np,u_net)
    plt.title("T="+str(T))
    plt.show()
    
def plotTrained3dAnim(u, T, d=2, n_steps=20, base_dir="", device="cpu"):
    assert d==2, "visualization is only implemented for 2-dimensional PDE"
    ts = torch.linspace(0,T,n_steps+1, device=device)
    with torch.no_grad():
        x0 = torch.linspace(-2,2,500)
        x1 = torch.linspace(-2,2,500)
        X0,X1 = torch.meshgrid([x0,x1])
        X = torch.cat([X0.reshape(-1,1), X1.reshape(-1,1)],1)
        t_coarse = ts[::n_steps//10]
        X = X.unsqueeze(1).repeat(1,len(t_coarse),1)
        t = t_coarse.reshape(1,-1,1).repeat(X.shape[0],1,1)
        shape_for_Y = t.shape 
        X = X.reshape(X.size(0) * X.size(1), -1)
        t = t.reshape(t.size(0) * t.size(1), -1)
        Y = u(t, X)
        Y = Y.reshape(shape_for_Y)

    ims = []
    fig = plt.figure()
    X0 = X0.numpy()
    X1 = X1.numpy()
    for idx, t in enumerate(t_coarse):
        Z = Y[:,idx,:].numpy().reshape(500,500) 
        im = plt.contourf(X0,X1,Z,levels=80,vmin=0,vmax=1)
        ims.append(im.collections)
        #plt.savefig(os.path.join(base_dir, "contourf{}.png".format(idx)))
    anim = animation.ArtistAnimation(fig, ims, interval=400, repeat_delay=3000)
    anim.save(os.path.join(base_dir, "contourf.mp4")) 
    anim.save(os.path.join(base_dir, "contourf.gif"), dpi=80, writer='imagemagick') 


def plotTrained3d(u, T, d=2, n_steps=20, base_dir="", base_name="eq", device="cpu"):
    assert d==2, "visualization is only implemented for 2-dimensional PDE"
    
    min_x = -1; max_x = 1
    min_y = -1; max_y = 1
    steps_xy = 100
    
    ts = torch.linspace(0,T,n_steps+1, device=device)
    with torch.no_grad():
        x0 = torch.linspace(min_x,max_x,steps_xy)
        x1 = torch.linspace(min_y,max_y,steps_xy)
        X0,X1 = torch.meshgrid([x0,x1])
        X = torch.cat([X0.reshape(-1,1), X1.reshape(-1,1)],1)
        t_coarse = ts[::n_steps//10]
        X = X.unsqueeze(1).repeat(1,len(t_coarse),1)
        shape_for_X = X.shape
        t = t_coarse.reshape(1,-1,1).repeat(X.shape[0],1,1)
        shape_for_Y = t.shape 
        X = X.reshape(X.size(0) * X.size(1), -1)
        t = t.reshape(t.size(0) * t.size(1), -1)
        Y = u(t, X)
        Y = Y.reshape(shape_for_Y)
        t = t.reshape(shape_for_Y)
        X = X.reshape(shape_for_X)
    
    X0 = X0.numpy()
    X1 = X1.numpy()
    for idx, t in enumerate(t_coarse):
        Z = Y[:,idx,:].numpy().reshape(steps_xy,steps_xy) 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X0, X1, Z, cmap='coolwarm',zorder=100)
        ax.set_zlim(-0.5, 4 )
        # ax.view_init(elev=20, azim=30)
        # ax.set_ylim(-3, 3)
        plt.title("T="+str(t.item()))
        plt.savefig(base_name+str(idx)+".pdf")
        plt.close()


def plotTrained3dContour(u, T, d=2, n_steps=20, base_dir="", base_name="eq", device="cpu"):
    assert d==2, "visualization is only implemented for 2-dimensional PDE"
    
    min_x = -2; max_x = 2
    min_y = -2; max_y = 2
    
    steps_xy = 100

    ts = torch.linspace(0,T,n_steps+1, device=device)
    with torch.no_grad():
        x0 = torch.linspace(min_x,max_x,steps_xy)
        x1 = torch.linspace(min_y,max_y,steps_xy)
        X0,X1 = torch.meshgrid([x0,x1])
        X = torch.cat([X0.reshape(-1,1), X1.reshape(-1,1)],1)
        t_coarse = ts[::n_steps//10]
        X = X.unsqueeze(1).repeat(1,len(t_coarse),1)
        shape_for_X = X.shape
        t = t_coarse.reshape(1,-1,1).repeat(X.shape[0],1,1)
        shape_for_Y = t.shape 
        X = X.reshape(X.size(0) * X.size(1), -1)
        t = t.reshape(t.size(0) * t.size(1), -1)
        Y = u(t, X)
        Y = Y.reshape(shape_for_Y)
        t = t.reshape(shape_for_Y)
        X = X.reshape(shape_for_X)
    
    X0 = X0.numpy()
    X1 = X1.numpy()
    for idx, t in enumerate(t_coarse):
        Z = Y[:,idx,:].numpy().reshape(steps_xy,steps_xy) 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.contourf(X0,X1,Z,levels=80,vmin=-4,vmax=0.5)
        ax.contourf(X0, X1, Z, levels=np.arange(-1,4,0.2), cmap='coolwarm')
        # ax.set_ylim(-3, 3)
        plt.title("T="+str(t.item()))
        plt.savefig(base_name+str(idx)+".pdf")
        plt.close()
