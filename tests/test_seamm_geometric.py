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


def test_the_gradient_criteria_reach_geomeTRIC_in_atomic_units(tmp_path, monkeypatch):
    """geomeTRIC reads its gradient criteria in E_h/bohr and its step criteria in Å,
    so criteria given in any units arrive converted to those."""
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
    criteria = [
        "Energy change criterion",
        "RMS gradient criterion",
        "Maximum atomic gradient criterion",
        "RMS step criterion",
        "Maximum atomic step criterion",
    ]

    class Step(geomeTRIC_mixin):
        directory = str(tmp_path)
        logger = logging.getLogger("test")
        metadata = {"convergence formulas": {"E+grad+step": {"criteria": criteria}}}

        def __init__(self):
            self._results = {}

        def get_system_configuration(self, P=None):
            return None, configuration

    captured = {}

    class Captured(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured()

    monkeypatch.setattr(geometric.optimize, "run_optimizer", capture)
    P = {
        "coordinate system": "TRIC",
        "max steps": 10,
        "calculate hessian": "never",
        "target": "minimum",
        "continue if not converged": "no",
        "convergence formula": "E+grad+step",
        "on error": "keep all subdirectories",
        "Energy change criterion": Q_(1e-6, "E_h"),
        "RMS gradient criterion": Q_(0.001, "eV/Å"),
        "Maximum atomic gradient criterion": Q_(0.0015, "eV/Å"),
        "RMS step criterion": Q_(1.2e-3, "Å"),
        "Maximum atomic step criterion": Q_(1.8e-3, "a_0"),
    }
    with pytest.raises(Captured):
        Step().run_geomeTRIC_optimizer(P, dict(P))

    eV_per_A = 1 / 51.42208619083232  # E_h/bohr in eV/Å
    assert captured["convergence_gmax"] == pytest.approx(0.0015 * eV_per_A)
    assert captured["convergence_grms"] == pytest.approx(0.001 * eV_per_A)
    assert captured["convergence_drms"] == pytest.approx(1.2e-3)
    assert captured["convergence_dmax"] == pytest.approx(1.8e-3 * 0.529177210903)
    assert captured["convergence_energy"] == pytest.approx(1e-6)
