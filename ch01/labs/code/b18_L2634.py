import torch

def pde_residual(net, x, t):
    """Residual of ∂u/∂t - D ∂²u/∂x²"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = net(x, t)
    
    # First-order derivatives via autograd
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    
    # Second-order derivative
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    D = 0.1
    return u_t - D * u_xx

# 학습 시 이 residual의 mean square가 손실에 더해지고,
# 그 손실이 다시 net의 파라미터에 대해 미분된다 (autograd가 자동 처리).