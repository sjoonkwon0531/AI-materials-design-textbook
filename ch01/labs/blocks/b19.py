# r: tensor of shape (N_atoms, 3), requires_grad=True
E = energy_model(r)
F = -torch.autograd.grad(E, r)[0]  # forces, shape (N_atoms, 3)
