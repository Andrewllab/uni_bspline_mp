
import torch

from ..phase_gn import LinearPhaseGenerator
from ..basis_gn import UniBSplineBasis
from .uni_bspline import UniformBSpline


class MPFactory:
    @staticmethod
    def init_mp(mp_type: str,
                mp_args: dict,
                num_dof: int = 1,
                tau: float = 3,
                delay: float = 0,
                learn_tau: bool = False,
                learn_delay: bool = False,
                dtype: torch.dtype = torch.float32,
                device: torch.device = "cpu"):
        """
        This is a helper class to initialize MPs,
        You can also directly initialize the MPs without using this class

        Create an MP instance given configuration

        Args:
            mp_type: type of movement primitives
            mp_args: arguments to a specific mp, refer each MP class
            num_dof: the number of degree of freedoms
            tau: default length of the trajectory
            delay: default delay before executing the trajectory
            learn_tau: if the length is a learnable parameter
            learn_delay: if the delay is a learnable parameter
            dtype: data type of the torch tensor
            device: device of the torch tensor


        Returns:
            MP instance
        """
        if mp_type == "uni_bspline":
            phase_gn = LinearPhaseGenerator(tau=tau, delay=delay,
                                            learn_tau=learn_tau,
                                            learn_delay=learn_delay,
                                            dtype=dtype, device=device)
            basis_gn = UniBSplineBasis(phase_generator=phase_gn,
                                       dtype=dtype, device=device,
                                       **mp_args)
            mp = UniformBSpline(basis_gn=basis_gn, num_dof=num_dof,
                                dtype=dtype, device=device, **mp_args)

        else:
            raise NotImplementedError

        return mp
