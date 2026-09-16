import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    x = torch.as_tensor(x0, dtype=torch.float32).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [x],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )

    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = f(x)
        loss.backward()
        optimizer.step()

    return x.detach()