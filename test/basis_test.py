import torch

from uniform_bs.basis_gn import LinearPhaseGenerator
from uniform_bs.basis_gn import UniBSplineBasis
import matplotlib.pyplot as plt

ph_gn = LinearPhaseGenerator()
b_basis = UniBSplineBasis(ph_gn, degree_p=5)
b_basis.show_basis(plot=True)
times = torch.linspace(-0.5, 1.5, 1000)
bs = b_basis.basis(times)
vb = b_basis.vel_basis(times)
ab = b_basis.acc_basis(times)

for i in range(b_basis.num_ctrlp-2):
    plt.plot(ab[:, i].numpy())
plt.show()
