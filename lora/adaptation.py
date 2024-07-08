import torch


def adapt_loss_fn(Ax, y):
    residual = Ax - y
    if Ax.dtype == torch.complex64:
        residual = torch.view_as_real(residual)
    loss = torch.mean(residual.pow(2))
    return loss