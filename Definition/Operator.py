import numpy as np
import torch


def GradOperator(x, y, grad_outputs=None, create_graph=False):
    """
    梯度算子.
    - 输入:
        - x: (M,D) 自变量;
        - y: (M,1) 因变量;
        - grad_outputs: (M,1) 因变量加权, 默认全一;
        - create_graph: 布尔值, 是否创建计算图, 计算高阶导数时应取 True.
    - 输出:
        - grad: (M,D) dy/dx 一阶导数/梯度.
    """
    assert type(x) == torch.Tensor, '自变量数据类型错误.'
    assert type(y) == torch.Tensor, '因变量数据类型错误.'
    assert x.requires_grad == True, '自变量需要保留梯度.'
    
    if grad_outputs is None:
        # 若权重为空, 则多输出时求导为 d sum(1*y_i)/dx
        grad_outputs = torch.ones_like(y)
    try:
        grad = torch.autograd.grad(
            outputs      = y,
            inputs       = x,
            grad_outputs = grad_outputs,
            create_graph = create_graph,
            retain_graph = True)[0]
    except:
        # 若求导失败, 则梯度应为零
        grad = torch.zeros_like(x)
    return grad

def InnerProductOperator(x1, x2, dim=0):
    """
    内积算子 (·).
    - 输入:
        - x1: (M,D) 向量 1;
        - x2: (M,D) 向量 2;
        - dim: 轴方向.
    - 输出:
        - output: (1,D)/(M,1) 向量内积
    """
    assert type(x1) == type(x2), '数据类型不一致.'
    assert x1.shape == x2.shape, '数据尺寸不一致.'

    if type(x1) == np.ndarray:
        output = np.sum(x1 * x2, axis=dim, keepdims=True)
    elif type(x1) == torch.Tensor:
        output = torch.sum(x1 * x2, dim=dim, keepdim=True)
    return output

def OuterProductOperator(x1, x2):
    """
    外积算子 (×).
    两个向量的叉乘/外积/矢量积仅在三维空间中有定义
    - 输入:
        - x1: (M,3) 向量 1;
        - x2: (M,3) 向量 2;

    - 输出:
        - output: (M,3) 外积向量.
    """
    assert type(x1) == type(x2), '数据类型不一致.'
    assert x1.shape == x2.shape, '数据尺寸不一致.'
    assert x1.shape[1] == 3, '叉积仅对三维空间有效.'

    res_list = [x1[:,1]*x2[:,2]-x1[:,2]*x2[:,1], x1[:,2]*x2[:,0]-x1[:,0]*x2[:,2], x1[:,0]*x2[:,1]-x1[:,1]*x2[:,0]]
    if type(x1) == np.ndarray:
        output = np.vstack(res_list).T
    elif type(x1) == torch.Tensor:
        output = torch.vstack(res_list).T
    return output

def DivOperator(x, y):
    """
    散度算子 (▽·).
    - 输入:
        - x: (M,D) 自变量;
        - y: (M,D) 因变量;
    - 输出:
        - div: (M,1) 散度向量, sum(dyi/dxi).
    """ 
    assert type(x) == torch.Tensor, '自变量数据类型错误.'
    assert type(y) == torch.Tensor, '因变量数据类型错误.'
    assert x.requires_grad == True, '自变量需要保留梯度.'
    assert x.shape == y.shape, '数据尺寸不一致'
    
    dim = x.shape[1]
    div = torch.zeros_like(x[:,0]).reshape(-1,1)
    for i in range(dim):
        div = div + (GradOperator(x, y[:,i], create_graph=True)[:,i]).reshape(-1,1)
    return div

def CurlOperator(x, y):
    """
    旋度算子 (▽×).
    - 输入:
        - x: (M,D) 自变量;
        - y: (M,D) 因变量;
    - 输出:
        - curl: (M,D) 旋度向量.
    """
    assert type(x) == torch.Tensor, '自变量数据类型错误.'
    assert type(y) == torch.Tensor, '因变量数据类型错误.'
    assert x.requires_grad == True, '自变量需要保留梯度.'
    assert x.shape == y.shape, '数据尺寸不一致.'
    assert x.shape[1] == 3, '旋度仅对三维空间有效.'

    P = GradOperator(x, y[:,0], create_graph=True)
    Q = GradOperator(x, y[:,1], create_graph=True)
    R = GradOperator(x, y[:,2], create_graph=True)
    # R_y-Q_z; P_z-R_x; Q_x-P_y
    curl = torch.vstack([R[:,1]-Q[:,2], P[:,2]-R[:,0], Q[:,0]-P[:,1]]).T
    return curl

def HessianOperator(x, y):
    """
    Hessian 算子
    """
    assert type(x) == torch.Tensor, '自变量数据类型错误.'
    assert type(y) == torch.Tensor, '因变量数据类型错误.'
    assert x.requires_grad == True, '自变量需要保留梯度.'
    
    grad = GradOperator(x, y, create_graph=True)
    dim = x.shape[1]
    hessian = torch.hstack([GradOperator(x, grad[:,i]) for i in range(dim)])
    hessian = hessian.reshape(-1, dim, dim)
    return hessian


# def PoissonOpt(x, y):
#     # 线性算子: Laplace/Poisson 方程
#     output = DivOperator(x, GradOperator(x, y, create_graph=True))
#     return output

# def HelmholtzOpt(x, y, k):
#     # 线性算子: Helmholtz 方程
#     output = DivOperator(x, GradOperator(x, y, create_graph=True)) + k**2 * y
#     return output

# def HeatOpt(x, y, alpha):
#     # 线性方程: 热方程/傅里叶方程 u_t-a u_xx
#     Dy = GradOperator(x, y, create_graph=True)
#     Dy_t = Dy[:,-1].reshape(-1,1) # 最后一维为时间维度
#     Laplace_x = torch.zeros_like(x[:,0]).reshape(-1,1)
#     for i in range(x.shape[1]-1):
#         Laplace_x = Laplace_x + GradOperator(x, Dy[:,i], create_graph=True)[:,i].reshape(-1,1)
#     output = Dy_t - alpha * Laplace_x
#     return output

# def WaveOpt(x, y, alpha):
#     # 线性方程: 波动方程 u_tt-a^2 u_xx
#     Dy = GradOperator(x, y, create_graph=True)
#     Dy_t = Dy[:,-1].reshape(-1,1) # 最后一维为时间维度
#     Laplace_x = torch.zeros_like(x[:,0]).reshape(-1,1)
#     for i in range(x.shape[1]-1):
#         Laplace_x = Laplace_x + GradOperator(x, Dy[:,i], create_graph=True)[:,i].reshape(-1,1)
#     Dy_tt = GradOperator(x, Dy_t, create_graph=True)[:,-1].reshape(-1,1)
#     output = Dy_tt - alpha ** 2 * Laplace_x
#     return output
