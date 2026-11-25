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


def _spring_step(tmp_path, P):
    """A geomeTRIC step whose engine is three harmonic springs (a bent triatomic),
    returning the step and the list of gradients the engine returned (E_h/bohr)."""
    import logging
    from types import SimpleNamespace

    import numpy as np
    from seamm_util import Q_

    from seamm_geometric import geomeTRIC_mixin

    bohr = Q_(1.0, "Å").m_as("a_0")
    # The minimum: O-H 0.96 Å, H-O-H 104.5°; springs on all three pairs
    theta = np.radians(104.5)
    minimum = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.96 * np.cos(theta), 0.96 * np.sin(theta), 0.0],
        ]
    )
    pairs = [(0, 1), (0, 2), (1, 2)]
    r0 = [np.linalg.norm(minimum[i] - minimum[j]) * bohr for i, j in pairs]
    k = 0.5  # E_h/bohr²

    start = np.array([[0.0, 0.0, 0.0], [1.05, 0.02, 0.0], [-0.30, 0.90, 0.05]])
    xyz = {"current": start.copy()}

    def set_coordinates(coordinates, fractionals=False):
        xyz["current"] = np.array(coordinates)

    atoms = SimpleNamespace(
        symbols=["O", "H", "H"],
        get_coordinates=lambda **kw: start.copy(),
        set_coordinates=set_coordinates,
    )
    configuration = SimpleNamespace(
        n_atoms=3, atoms=atoms, coordinates=start.tolist(), bonds=[]
    )
    criteria = [
        "Energy change criterion",
        "RMS gradient criterion",
        "Maximum atomic gradient criterion",
        "RMS step criterion",
        "Maximum atomic step criterion",
    ]
    engine_gradients = []

    class Step(geomeTRIC_mixin):
        directory = str(tmp_path)
        logger = logging.getLogger("test")
        metadata = {"convergence formulas": {"E+grad+step": {"criteria": criteria}}}
        references = SimpleNamespace(cite=lambda **kw: None)

        def __init__(self):
            self._results = {}
            self._bibliography = {}

        def get_system_configuration(self, P=None):
            return None, configuration

        def geometric_calculate_gradients(self, coordinates):
            """Energy (E_h) and gradients (E_h/bohr) of the springs at the
            coordinates (Å)."""
            x = np.asarray(coordinates) * bohr
            energy = 0.0
            gradients = np.zeros_like(x)
            for (i, j), r_eq in zip(pairs, r0):
                d = x[i] - x[j]
                r = np.linalg.norm(d)
                energy += 0.5 * k * (r - r_eq) ** 2
                g = k * (r - r_eq) * d / r
                gradients[i] += g
                gradients[j] -= g
            engine_gradients.append(gradients.copy())
            return energy, gradients

    P = {
        "coordinate system": "TRIC",
        "calculate hessian": "never",
        "target": "minimum",
        "continue if not converged": "no",
        "convergence formula": "E+grad+step",
        "on error": "keep all subdirectories",
        "on success": "keep all subdirectories",
        "Energy change criterion": Q_(1e-6, "E_h"),
        "RMS gradient criterion": Q_(3e-4, "E_h/a_0"),
        "Maximum atomic gradient criterion": Q_(4.5e-4, "E_h/a_0"),
        "RMS step criterion": Q_(1.2e-3, "a_0"),
        "Maximum atomic step criterion": Q_(1.8e-3, "a_0"),
        **P,
    }
    return Step(), P, engine_gradients


def _max_and_rms_kJ_per_mol_A(gradients):
    import numpy as np
    from seamm_util import Q_

    norms = np.linalg.norm(gradients, axis=1) * Q_(1.0, "E_h/a_0").m_as("kJ/mol/Å")
    return norms.max(), np.sqrt(np.mean(norms**2))


def test_the_final_forces_are_reported_in_kJ_per_mol_per_A(tmp_path):
    """After a converged optimization the maximum and RMS forces are those of the
    engine's last gradient, in kJ/mol/Å (not geomeTRIC's E_h/bohr)."""
    step, P, engine_gradients = _spring_step(tmp_path, {"max steps": 50})
    step.run_geomeTRIC_optimizer(P, dict(P))

    assert step._results["converged"]
    max_force, rms_force = _max_and_rms_kJ_per_mol_A(engine_gradients[-1])
    assert max_force > 0.0
    assert step._results["maximum_gradient"] == pytest.approx(max_force, rel=1e-6)
    assert step._results["rms_gradient"] == pytest.approx(rms_force, rel=1e-6)
    # Converged, so below the criterion, which is in E_h/bohr
    assert step._results["maximum_gradient"] < 4.5e-4 * 4961.47
    # The last step is in Å, and below the criterion of 1.8e-3 bohr
    assert 0.0 < step._results["maximum_step"] < 1.8e-3 * 0.529177210903


def test_the_forces_of_an_unconverged_optimization_are_in_kJ_per_mol_per_A(tmp_path):
    """An optimization that stops at its maximum number of steps still reports the
    last forces, in kJ/mol/Å, before raising."""
    step, P, engine_gradients = _spring_step(tmp_path, {"max steps": 2})
    with pytest.raises(RuntimeError, match="did not converge"):
        step.run_geomeTRIC_optimizer(P, dict(P))

    assert not step._results["converged"]
    max_force, rms_force = _max_and_rms_kJ_per_mol_A(engine_gradients[-1])
    assert step._results["maximum_gradient"] == pytest.approx(max_force, rel=1e-6)
    assert step._results["rms_gradient"] == pytest.approx(rms_force, rel=1e-6)
