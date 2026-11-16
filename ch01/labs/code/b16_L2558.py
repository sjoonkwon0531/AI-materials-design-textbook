import torch

# Tensor를 grad가 필요한 leaf로 선언
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum() + torch.sin(x[0])

# Backward pass로 그래디언트 계산
y.backward()

print(x.grad)  # ∂y/∂x = [2x_1 + cos(x_1), 2x_2, 2x_3]
# 출력: tensor([2.5403, 4.0000, 6.0000])