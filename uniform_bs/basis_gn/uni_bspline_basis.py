
import torch

# from mp_pytorch.phase_gn import PhaseGenerator
from .basis_generator import BasisGenerator
from ..phase_gn import LinearPhaseGenerator


class UniBSplineBasis(BasisGenerator):

    def __init__(self,
                 phase_generator: LinearPhaseGenerator,
                 num_basis: int = 10,
                 degree_p: int = 3,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu'):

        super().__init__(phase_generator, num_basis, dtype, device)
        self.degree_p = degree_p

        # todo check how the velocities at the beginning and end look like
        # todo padding at the end, when the time exceed 1
        # todo check how to set knots, when I want to start from given init conds
        # number of knots needed, with respect to B-sp degree and number of
        # control points
        num_knots = self.degree_p + 1 + num_basis
        num_knots_non_rep_inside_1 = num_knots - 2*self.degree_p
        # uniform knots vector
        self.knots_vec = torch.linspace(0, 1, num_knots_non_rep_inside_1,
                                        dtype=dtype, device=device)
        knots_prev = torch.zeros(self.degree_p, dtype=dtype, device=device)
        knots_pro = torch.ones(self.degree_p, dtype=dtype, device=device)
        self.knots_vec = torch.cat([knots_prev, self.knots_vec, knots_pro])

    def basis(self, times: torch.Tensor) -> torch.Tensor:

        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of basis:
        # [*add_dim, num_times, num_basis]

        phase = self.phase_generator.phase(times)

        basis = [self._basis_function(i, self.degree_p, self.knots_vec, phase)
                 for i in range(self.num_basis)]
        basis = torch.stack(basis, dim=-1)

        return basis

    def _basis_function(self, i, k, knots, u, **kwargs):
        """
        recursive construct of B-spline basis using de Boor's algorithm

        :param i: basis index
        :param k: degree
        :param u: evaluate time point
        :param knots: knots vector
        :return: vector of shape [num_eval_points]
        """

        # adding some assertion to tell whether i is a feasible choice

        if k == 0:
            num_basis = kwargs.get("num_basis", self.num_basis)
            if i == num_basis-1:
                # with regard to original definition, each span is defined as \
                # left closed and right open interval [v_i, v_i+1), which makes\
                # the value at right end always 0. It is undesired,so that we \
                # need to handle the last basis specially
                b0 = torch.where((u >= knots[i]) & (u <= knots[i+1]), 1, 0)
            else:
                b0 = torch.where((u >= knots[i]) & (u < knots[i+1]), 1, 0)
            return torch.as_tensor(b0, dtype=self.dtype, device=self.device)
        else:
            denom1 = knots[i + k] - knots[i]
            term1 = 0.0 if denom1 == 0 else (u - knots[i]) / denom1 * \
                self._basis_function(i, k - 1, knots, u, **kwargs)
            denom2 = knots[i + k + 1] - knots[i + 1]
            term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - u) / denom2 * \
                self._basis_function(i + 1, k - 1, knots, u, **kwargs)
            return term1 + term2

    def velocity_control_points(self, ctrl_pts: torch.Tensor):
        """
        given the position control points (parameter), return the velocity control
        points for vel B-spline as linear combination of position control points.

        :param ctrl_pts:
        :return:
        """
        # todo 想做constraint的话，应该写成矩阵乘法
        # diff shape: [*add_dim, num_dof, num_basis-1]
        diff = ctrl_pts[..., 1:] - ctrl_pts[..., :-1]
        # shape: [num_basis-1]
        delta = self.knots_vec[1+self.degree_p: self.num_basis+self.degree_p] -\
                self.knots_vec[1: self.num_basis]  # todo check formula
        #一般情况，注意公式是0-n共n+1个ctr_points
        diff = diff * (1/delta)  # todo broadcast可能不对，需要torch.einsum
        return diff * self.degree_p

    def acceleration_control_points(self, ctrl_pts: torch.Tensor):
        """
        given the position control points (parameter), return the acceleration
        control points for acc B-spline as linear combination of position
        control points.

        :param ctrl_pts:
        :return:
        """
        # shape: [*add_dim, num_dof, num_basis-1]
        vel_ctrl_pts = self.velocity_control_points(ctrl_pts)
        # shape: [*add_dim, num_dof, num_basis-2]
        diff = vel_ctrl_pts[..., 1:] - ctrl_pts[..., :-1]
        # shape: [num_basis-2]
        delta = self.knots_vec[2+self.degree_p: self.num_basis+self.degree_p-1]\
            - self.knots_vec[2: self.num_basis-1]
        diff = diff * (1/delta)
        return diff * (self.degree_p-1)

    # def velocity_basis(self, times: torch.Tensor) -> torch.Tensor:
    #     # not correct
    #     phase = self.phase_generator.phase(times)
    #
    #     basis = [self._basis_function(i, self.degree_p-1, self.knots_vec, phase)
    #              for i in range(1, self.num_basis-1)]
    #     basis = torch.stack(basis, dim=-1)
    #
    #     return basis

    def vel_basis(self,times: torch.Tensor) -> torch.Tensor:

        phase = self.phase_generator.phase(times)

        # for clamped uni B-spline
        vel_nots_vec = self.knots_vec[1:-1]
        basis = \
            [self._basis_function(i, self.degree_p-1, vel_nots_vec, phase, num_basis=self.num_basis-1)
             for i in range(self.num_basis-1)]
        basis = torch.stack(basis, dim=-1)
        return basis

    # def acceleration_basis(self, times: torch.Tensor) -> torch.Tensor:
    #
    #     phase = self.phase_generator.phase(times)
    #
    #     basis = [
    #         self._basis_function(i, self.degree_p - 2, self.knots_vec, phase)
    #         for i in range(2, self.num_basis - 2)]
    #     basis = torch.stack(basis, dim=-1)
    #
    #     return basis

    def acc_basis(self, times: torch.Tensor) -> torch.Tensor:

        phase = self.phase_generator.phase(times)
        acc_knots_vec = self.knots_vec[2: -2]

        basis = [
            self._basis_function(i, self.degree_p - 2, acc_knots_vec, phase, num_basis=self.num_basis-2)
            for i in range(0, self.num_basis - 2)]
        basis = torch.stack(basis, dim=-1)

        return basis




