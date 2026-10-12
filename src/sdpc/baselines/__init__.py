"""Baseline controllers (MPC via CasADi/IPOPT)."""
from .mpc import solve_mpc, mpc_available

__all__ = ["solve_mpc", "mpc_available"]
