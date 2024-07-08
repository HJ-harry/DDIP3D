import torch
from fastmri_utils import fft2_m, ifft2_m, ifft2c_new, fft2c_new



class SinglecoilMRI_real:
    def __init__(self, mask):
        self.mask = mask

    def _A(self, x):
        return fft2_m(x) * self.mask

    def _Adagger(self, x):
        return torch.real(ifft2_m(x))

    def _AT(self, x):
        return self._Adagger(x)


class SinglecoilMRI_comp:
    def __init__(self, mask):
        self.mask = mask

    def _A(self, x):
        return fft2_m(x) * self.mask

    def _Adagger(self, x):
        return ifft2_m(x)

    def _AT(self, x):
        return self._Adagger(x)


class MulticoilMRI():
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def _A(self, x, mps):
        return fft2_m(mps * x) * self.mask
    
    def _Adagger(self, x, mps):
        return torch.sum(torch.conj(mps) * ifft2_m(x * self.mask), dim=1).unsqueeze(dim=1)

    def _AT(self, x, mps):
        return self._Adagger(x, mps)

    