"""
Unit and regression test for the seamm_ase package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest  # noqa: F401

import seamm_geometric  # noqa: F401


def test_seamm_ase_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "seamm_geometric" in sys.modules


def test_the_error_that_stopped_the_optimization_is_raised(tmp_path, monkeypatch):
    """An error in geomeTRIC (e.g. an engine that could not start) is raised as
    itself, not as an UnboundLocalError or "did not converge"."""
    import logging
    from types import SimpleNamespace

    import geometric.optimize
    import numpy as np
    from seamm_util import Q_

    from seamm_geometric import geomeTRIC_mixin

    xyz = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    atoms = SimpleNamespace(
        symbols=["O", "H", "H"], get_coordinates=lambda **kw: xyz.copy()
    )
    configuration = SimpleNamespace(
        n_atoms=3, atoms=atoms, coordinates=xyz.tolist(), bonds=[]
    )

    class Step(geomeTRIC_mixin):
        directory = str(tmp_path)
        logger = logging.getLogger("test")
        metadata = {"convergence formulas": {"E+grad": {"criteria": []}}}

        def __init__(self):
            self._results = {}

        def get_system_configuration(self, P=None):
            return None, configuration

    def fail(**kwargs):
        raise RuntimeError("MDI Error: MDI_Send_Command failed")

    monkeypatch.setattr(geometric.optimize, "run_optimizer", fail)
    P = {
        "coordinate system": "TRIC",
        "max steps": 10,
        "calculate hessian": "never",
        "target": "minimum",
        "continue if not converged": "no",
        "convergence formula": "E+grad",
        "on error": "keep all subdirectories",
        "Energy change criterion": Q_(1e-6, "E_h"),
    }
    with pytest.raises(RuntimeError, match="MDI_Send_Command failed"):
        Step().run_geomeTRIC_optimizer(P, dict(P))
