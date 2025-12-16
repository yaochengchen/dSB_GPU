"""
Simulated bifurcation (SB) algorithms and its variants.

Final cleaned version (only QAIA / SB / CDSB).
"""

import numpy as np
import torch

import sys
from pathlib import Path

# 获取当前文件的上级目录（..）的绝对路径
parent_path = Path(__file__).resolve().parent.parent

# 添加到 sys.path
sys.path.append(str(parent_path))

#from MultiRowSharedFused import MultiRowSharedFused


class QAIA:
    """
    The base class of QAIA.

    Args:
        J: coupling matrix, shape (N, N). torch dense or torch sparse.
        h: external field, shape (N,) or (N,1). Default: None.
        x: initialized spin values, shape (N, batch_size). Default: None.
        n_iter: number of iterations. Default: 1000.
        batch_size: number of sampling. Default: 1.
        device: 'cuda' or 'cpu'. Default: 'cuda'.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, device="cuda"):
        # ensure sparse CSR on device
        if J.layout != torch.sparse_csr:
            J = J.to(device)
            J = J.to_sparse_csr()
        else:
            J = J.to(device)

        self.J = J
        self.device = device

        # h: keep as (N,1) if provided
        if h is not None:
            h = h.to(device)
            if h.dim() == 1:
                h = h.view(-1, 1)
            self.h = h
        else:
            self.h = None

        self.x = x
        self.N = self.J.shape[0]
        self.n_iter = n_iter
        self.batch_size = batch_size

    def initialize(self):
        """Randomly initialize spin values."""
        self.x = 0.02 * (torch.rand(self.N, self.batch_size, device=self.device) - 0.5)

    def calc_cut(self, x=None):
        """
        Calculate cut value.

        Args:
            x: spin values (N, batch_size). If None, use self.x.
        """
        if x is None:
            sign = torch.sign(self.x)
        else:
            sign = torch.sign(x)

        return (
            0.25 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0)
            - 0.25 * self.J.sum()
        )

    def calc_energy(self, x=None):
        """
        Calculate energy.

        Args:
            x: spin values (N, batch_size). If None, use self.x.
        """
        if x is None:
            sign = torch.sign(self.x)
        else:
            sign = torch.sign(x)

        if self.h is None:
            return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0)

        # match screenshot behavior: keepdim=True then subtract torch.mm(h.T, sign)
        return (
            -0.5
            * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0, keepdim=True)
            - torch.mm(self.h.T.type(torch.float32), sign)
        )


class SB(QAIA):
    """
    Base class of SB.

    Adds:
        delta, dt, p (pumping amplitude schedule), xi (frequency scale),
        plus initializes momentum y.

    Args are same as QAIA plus:
        dt: step size. Default: 1.
        xi: positive constant with dimension of frequency. Default: None.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        device="cuda",
        dt=1,
        xi=None,
    ):
        super().__init__(J, h, x, n_iter, batch_size, device)

        # positive detuning frequency
        self.delta = 1
        self.dt = dt

        # pumping amplitude
        self.p = np.linspace(0, 1, self.n_iter)
        self.xi = xi

        # xi default (match screenshot logic)
        if self.xi is None:
            if self.h is not None:
                self.xi = (
                    0.5
                    * np.sqrt(self.N - 1)
                    / torch.sqrt(
                        (self.J.to_dense() ** 2).sum()
                        + 2 * ((self.h / 2) ** 2).sum()
                    )
                )
            else:
                self.xi = (
                    0.5 * np.sqrt(self.N - 1) / torch.sqrt((self.J.to_dense() ** 2).sum())
                )

        self.x = x
        self.initialize()

    def initialize(self):
        """Initialize spin values and momentum."""
        if self.x is None:
            self.x = 0.02 * (torch.rand(self.N, self.batch_size, device=self.device) - 0.5)

        if self.x.shape[0] != self.N:
            raise ValueError(
                f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}"
            )

        self.y = 0.02 * (torch.rand(self.N, self.batch_size, device=self.device) - 0.5)


class CDSB(SB):
    """
    CDSB variant.

    Reference:
        High-performance combinatorial optimization based on classical mechanics
        https://www.science.org/doi/10.1126/sciadv.abe7953
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        device="cuda",
        dt=1,
        xi=None,
    ):
        """Construct DSB algorithm."""
        # ensure sparse CSR
        if J.layout != torch.sparse_csr:
            J = J.to(device)
            J = J.to_sparse_csr()
        else:
            J = J.to(device)

        self.J = J
        self.h = h
        self.device = device

        if self.h is not None:
            self.h = self.h.to(device)
            if self.h.dim() == 1:
                self.h = self.h.view(-1, 1)
            self.J = self.combine_tensor()  # returns sparse CSR

        self.x = x

        # The number of spins
        self.N = self.J.shape[0]
        self.n_iter = n_iter
        self.batch_size = batch_size

        # positive detuning frequency
        self.delta = 1
        self.dt = dt

        # pumping amplitude
        self.p = torch.linspace(0, 1, self.n_iter)
        self.xi = xi

        if self.xi is None:
            self.xi = (
                0.5 * np.sqrt(self.N - 1) / torch.sqrt((self.J.to_dense() ** 2).sum())
            )

        self.x = x
        super().initialize()

    def update(self):
        #multi_row_updater = MultiRowSharedFused(self.J, verbose=False)
        #self.y, self.x = multi_row_updater.run(
        #    self.y, self.x,
        #    self.delta, self.p, self.xi, self.dt, self.n_iter
        #)
        for i in range(self.n_iter):

            self.y += (
                -(self.delta - self.p[i]) * self.x + self.xi * (torch.sparse.mm(self.J, torch.sign(self.x)))
            ) * self.dt
            self.x += self.dt * self.y * self.delta
            cond = torch.abs(self.x) > 1.0
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)

    @staticmethod
    def symmetrize(tensor: torch.Tensor):
        return (tensor + tensor.t()) / 2.0

    def combine_tensor(self):
        """Gather self.J and self.h into a single matrix."""
        if self.h is None:
            return self.J

        dimen = self.J.shape[0]
        J_dense = self.symmetrize(self.J.to_dense())

        tensor = torch.zeros((dimen + 1, dimen + 1), device=self.device)
        tensor[:dimen, :dimen] = J_dense
        tensor[:dimen, dimen] = -self.h[:, 0]
        tensor[dimen, :dimen] = -self.h[:, 0]

        return tensor.to_sparse_csr()

    def calc_energy(self, x=None):
        """
        Calculate energy.

        Args:
            x: spin values (N, batch_size). If None, use self.x.
        """
        if x is None:
            sign = torch.sign(self.x)
        else:
            sign = torch.sign(x)

        return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0)

