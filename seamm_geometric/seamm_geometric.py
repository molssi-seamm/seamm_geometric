# -*- coding: utf-8 -*-

__all__ = ["geomeTRIC_mixin", "SEAMMEngine"]

import calendar
from datetime import datetime
from importlib.resources import files
import json
import logging
import math
import os
from pathlib import Path
import pprint
import re
import shutil
import string
import sys
import time
import traceback

import bibtexparser
import geometric
import geometric.molecule
import numpy as np
from tabulate import tabulate

from seamm_util import Q_
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __
from ._version import __version__  # noqa: F401

# In addition to the normal logger, two logger-like printing facilities are
# defined: "job" and "printer". "job" send output to the main job.out file for
# the job, and should be used very sparingly, typically to echo what this step
# will do in the initial summary of the job.
#
# "printer" sends output to the file "step.out" in this steps working
# directory, and is used for all normal output from this step.

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("goemeTRIC")

# Regexp to remove ansi escape sequences
ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = Path(newPath).expanduser()

    def __enter__(self):
        self.savedPath = Path.cwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class SEAMMEngine(geometric.engine.Engine):
    """Helper class that is a geomeTRIC engine and connects to SEAMM."""

    def __init__(self, step, molecule):
        """Initialize this geomeTRIC engine.

        Parameters
        ----------
        step : seamm.node
            The SEAMM plug-in using this.
        molecule : geometric.molecule.Molecule
            The geomeTRIC molecule to work with
        """
        self.step = step
        super().__init__(molecule)

    def calc_new(self, coords, dirname):
        """The method to calculate the new energy and forces.

        Parameters
        ----------
        coords : np.ndarray
            A 1-D Numpy array of the coordinates, in Bohr
        dirname : The name of a directory (not used)

        Returns
        -------
        data : {str : any}
            The result returned to geomeTRIC:

                "energy" : The energy, in Hartree

                "gradient" : The 1st derivative, or gradient, of the energy in
                Hartree/bohr as a 1-D Numpy array.
        """
        xyz = coords.reshape(-1, 3) * Q_(1.0, "a_0").m_as("angstrom")

        energy, gradient = self.step.calculate_gradients(xyz)

        return {"energy": energy, "gradient": gradient.ravel()}


class geomeTRIC_mixin:
    """A mixin class for running a geomeTRIC optimization."""

    def read_bibliography(self):
        """Read the bibliography from a file and add to the local bibliography"""
        if "geomeTRIC" not in self._bibliography:
            try:
                data = (
                    files("seamm_geometric.data").joinpath("references.bib").read_text()
                )
                tmp = bibtexparser.loads(data).entries_dict
                writer = bibtexparser.bwriter.BibTexWriter()
                for key, data in tmp.items():
                    self.logger.info(f"      {key}")
                    self._bibliography[key] = writer._entry_to_bibtex(data)
                self.logger.debug("Bibliography\n" + pprint.pformat(self._bibliography))
            except Exception:
                pass

    def calculate_gradients(self, coordinates):
        """Given the new coordinates, calculate the energy and gradients.

        Parameters
        ----------
        coordinates : [3, n_atoms] array of coordinates
        """
        self.step = self.step + 1
        fmt = "05d"

        # Make the geometric output readable by removing the ANSI escape sequences
        logPath = Path("geomeTRIC.out")
        if logPath.exists():
            text = logPath.read_text()
            text = ansi_escape.sub("", text)
            logPath.write_text(text)

        n_atoms = self._working_configuration.n_atoms

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("\nnew coordinates")
            for i in range(n_atoms):
                self.logger.debug(
                    f"   {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f} "
                    f"{coordinates[i][2]:8.3f}"
                )

        # Set the coordinates in the configuration
        self._working_configuration.atoms.set_coordinates(
            coordinates, fractionals=False
        )

        # Find the handler for job.out and set the level up
        job_handler = None
        out_handler = None
        for handler in job.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and "job.out" in handler.baseFilename
            ):
                job_handler = handler
                job_level = job_handler.level
                job_handler.setLevel(printing.JOB)
            elif isinstance(handler, logging.StreamHandler):
                out_handler = handler
                out_level = out_handler.level
                out_handler.setLevel(printing.JOB)

        # Get the first real node
        first_node = self.subflowchart.get_node("1").next()

        # Ensure the nodes have their options
        node = first_node
        while node is not None:
            node.all_options = self.all_options
            node = node.next()

        # And the subflowchart has the executor
        self.subflowchart.executor = self.flowchart.executor

        # Direct most output to iteration.out
        step_id = f"step_{self.step:{fmt}}"
        step_dir = Path(self._working_directory) / step_id
        step_dir.mkdir(parents=True, exist_ok=True)

        # A handler for the file
        if self._file_handler is not None:
            self._file_handler.close()
            job.removeHandler(self._file_handler)
        path = step_dir / "Step.out"
        path.unlink(missing_ok=True)
        self._file_handler = logging.FileHandler(path)
        self._file_handler.setLevel(printing.NORMAL)
        formatter = logging.Formatter(fmt="{message:s}", style="{")
        self._file_handler.setFormatter(formatter)
        job.addHandler(self._file_handler)

        # Add the step to the ids so the directory structure is reasonable
        self.subflowchart.reset_visited()
        name = self._working_directory.name
        self.set_subids((*self._id, name, step_id))

        # Run through the steps in the loop body
        node = first_node
        try:
            while node is not None:
                node = node.run()
        except DeprecationWarning as e:
            printer.normal("\nDeprecation warning: " + str(e))
            traceback.print_exc(file=sys.stderr)
            traceback.print_exc(file=sys.stdout)
        except Exception as e:
            printer.job(f"Caught exception in step {self.step}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            with open(step_dir / "stderr.out", "a") as fd:
                traceback.print_exc(file=fd)
            raise
        self.logger.debug(f"End of step {self.step}")

        # Remove any redirection of printing.
        if self._file_handler is not None:
            self._file_handler.close()
            job.removeHandler(self._file_handler)
            self._file_handler = None
        if job_handler is not None:
            job_handler.setLevel(job_level)
        if out_handler is not None:
            out_handler.setLevel(out_level)

        # Get the energy and derivatives
        paths = sorted(step_dir.glob("**/Results.json"))

        if len(paths) == 0:
            raise RuntimeError(
                "There are no energy and gradients in properties.json for step "
                f"{self.step} in {step_dir}."
            )
        else:
            # Find the most recent and assume that is the one wanted
            newest_time = None
            for path in paths:
                with path.open() as fd:
                    data = json.load(fd)
                time = datetime.fromisoformat(data["iso time"])
                if newest_time is None:
                    newest = path
                    newest_time = time
                elif time > newest_time:
                    newest_time = time
                    newest = path
            with newest.open() as fd:
                data = json.load(fd)

        energy = data["energy"]
        if "energy,units" in data:
            units = data["energy,units"]
        else:
            units = "kJ/mol"
        energy *= Q_(1.0, units).to("E_h").magnitude

        gradients = data["gradients"]

        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug("\ngradients")
            for i in range(n_atoms):
                logger.debug(
                    f"   {gradients[i][0]:8.3f} {gradients[i][1]:8.3f} "
                    f"{gradients[i][2]:8.3f}"
                )

        if "gradients,units" in data:
            funits = data["gradients,units"]
        else:
            funits = "kJ/mol/Å"

        # Get the measures of convergence
        max_force = np.max(np.linalg.norm(gradients, axis=1))
        self._data["max_force"].append(max_force)
        self._results["maximum_gradient"] = Q_(max_force, funits).m_as("kJ/mol/Å")
        rms_force = np.sqrt(np.mean(np.linalg.norm(gradients, axis=1) ** 2))
        self._data["rms_force"].append(rms_force)
        self._results["rms_gradient"] = Q_(rms_force, funits).m_as("kJ/mol/Å")

        if self._step > 1:
            step = coordinates - self._last_coordinates
            max_step = np.max(np.linalg.norm(step, axis=1))
        else:
            max_step = 0.0
        self._data["max_step"].append(max_step)
        self._results["maximum_step"] = max_step
        self._last_coordinates = np.array(coordinates)

        # Units!
        gradients = np.array(gradients) * Q_(1.0, funits).to("E_h/a_0").magnitude

        return energy, gradients

    def describe_geomeTRIC_optimizer(self, P=None, short=False, natoms=None):
        """Describe the geomeTRIC optimizer.

        Parameters
        ----------
        P : dict
            An optional dictionary of the current values of the control
            parameters.

        short : bool
            If True, return a short description of the step.

        natoms : int
            The number of atoms in the structure.

        Returns
        -------
        str
            A description of the current step.
        """
        if P is None:
            P = self.parameters.values_to_dict()

        target = P["target"]
        convergence = P["convergence"]
        convergence_formula = P["convergence formula"]
        formula = self.metadata["convergence formulas"][convergence_formula]

        if target == "minimum":
            text = "The structure will be optimized"
        elif target == "transition state":
            text = "The transition state will be optimized"
        elif self.is_expr(target):
            text = (
                "The structure will be optimized to a minimum, transition state, etc. "
                "depending on the value of '{target}'"
            )
        else:
            raise ValueError(f"Unknown target {target}")

        text += " using the geomeTRIC optimizer with {coordinate system}. "

        if self.is_expr(P["calculate hessian"]):
            text += (
                "'{calculate hessian}' will determine whether and how often to "
                "calculate the Hessian matrix."
            )
        elif P["calculate hessian"] != "never":
            text += (
                "The Hessian matrix will be calculated for {calculate hessian} step."
            )

        text += " Convergence will be reached when "
        tmp_text = ["\n"]
        criteria = (
            "energy change criterion",
            "rms gradient criterion",
            "atomic gradient criterion",
            "rms step criterion",
            "atomic step criterion",
        )
        if self.is_expr(convergence):
            text += "the conditions defined by '{convergence}' are met."
            text += " The formula may use any of the following criteria:"
        elif convergence == "custom":
            if convergence_formula == "E+grad+step":
                text += "the following conditions are met:"
            else:
                text += (
                    f"the conditions of {convergence_formula} are met with the "
                    "following custom values:"
                )
        elif convergence_formula == "MOPAC":
            text += f"the conditions of {convergence} are met using MOPAC's formula:"
            tmp_text.append("_norm of gradient_ < Gradient norm".center(70))
            criteria = formula["criteria"]
        else:
            text += f"the conditions for {convergence} are met:"
            for line in formula["text"].splitlines():
                tmp_text.append(line.center(70))
            criteria = formula["criteria"]

        result = str(__(text, dedent=False, indent=3 * " ", **P))
        result += "\n".join(tmp_text)

        table = {
            "Criterion": [],
            "Value": [],
            "Units": [],
        }
        for criterion in criteria:
            table["Criterion"].append(criterion[0:-10])
            table["Value"].append(self.parameters[criterion].value)
            table["Units"].append(self.parameters[criterion].units)

        tmp = tabulate(table, headers="keys", tablefmt="rounded_outline")

        result += "\n"
        result += str(__(tmp, indent=7 * " ", wrap=False))
        result += "\n"
        result += "\n"

        return result

    def run_geomeTRIC_optimizer(self, P, PP):
        """Run an optimization using geomeTRIC.

        Parameters
        ----------
        P : dict
            The current values of the parameters
        PP : dict
            The current values of the parameters, formatted for printing
        """
        self._data = {
            "step": [],
            "energy": [],
            "max_force": [],
            "rms_force": [],
            "max_step": [],
        }
        self._last_coordinates = None
        self._step = 0

        # Create the directory
        directory = Path(self.directory)
        self._working_directory = directory / "geomeTRIC"
        self._working_directory.mkdir(parents=True, exist_ok=True)

        _, starting_configuration = self.get_system_configuration()
        _, self._working_configuration = self.get_system_configuration(P)
        n_atoms = starting_configuration.n_atoms

        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug("initial coordinates")
            coordinates = starting_configuration.coordinates
            symbols = starting_configuration.atoms.symbols
            for i in range(n_atoms):
                logger.debug(
                    f"   {symbols[i]} {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f}"
                    f" {coordinates[i][2]:8.3f}"
                )
            logger.debug(starting_configuration.bonds)

        coordinates = self._working_configuration.atoms.get_coordinates(
            fractionals=False, as_array=True
        )

        geoMol = geometric.molecule.Molecule()
        geoMol.elem = self._working_configuration.atoms.symbols
        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug("coordinates")
            for i in range(n_atoms):
                logger.debug(
                    f"   {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f} "
                    f"{coordinates[i][2]:8.3f}"
                )

        geoMol.xyzs = [coordinates]

        customengine = SEAMMEngine(self, geoMol)

        coordsys = P["coordinate system"].split(":")[0].lower()

        max_steps = P["max steps"]
        if max_steps == "default":
            max_steps = 12 * n_atoms
        elif "natoms" in max_steps:
            tmp = max_steps.split()
            if "natoms" in tmp[0]:
                max_steps = int(tmp[1]) * n_atoms
            else:
                max_steps = int(tmp[0]) * n_atoms
        else:
            max_steps = P["max steps"]

        self.step = 0
        logPath = self._working_directory / "geomeTRIC.out"
        logIni = self._working_directory / "log.ini"
        logIni.write_text(
            f"""\
# The default logging configuration file for geomeTRIC
# Modified to write to {logPath}

[loggers]
keys=root

[handlers]
keys=file_handler

[formatters]
keys=formatter

[logger_root]
level=INFO
handlers=file_handler

[handler_file_handler]
class=geometric.nifty.RawFileHandler
level=INFO
formatter=formatter
args=("{logPath}",)

[formatter_formatter]
format=%(message)s
#format=%(asctime)s %(name)-12s %(levelname)-8s %(message)s
"""
        )
        cc_logger = logging.getLogger("cclib")
        cc_logger.setLevel(logging.WARNING)

        # Work out the convergence criteria, etc.
        kwargs = {
            "maxiter": max_steps,
            "hessian": P["calculate hessian"],
            "frequency": P["calculate hessian"] != "never",
            "transition": "transition" in P["target"].lower(),
            "coordsys": coordsys,
        }

        convergence_formulas = self.metadata["convergence formulas"]

        convergence_formula = P["convergence formula"]
        criteria = convergence_formulas[convergence_formula]["criteria"]

        # Set the convergence criteria to large value (so always met) and then
        # set the ones we want to use.
        kwargs["convergence_energy"] = 99.9
        kwargs["convergence_grms"] = 99.9
        kwargs["convergence_gmax"] = 99.9
        kwargs["convergence_drms"] = 99.9
        kwargs["convergence_dmax"] = 99.9

        if "Energy change criterion" in criteria:
            kwargs["convergence_energy"] = P["Energy change criterion"].m_as("E_h")
        if "RMS gradient criterion" in criteria:
            kwargs["convergence_grms"] = P["RMS gradient criterion"].m_as("E_h/Å")
        if "Maximum atomic gradient criterion" in criteria:
            kwargs["convergence_gmax"] = P["Maximum atomic gradient criterion"].m_as(
                "E_h/Å"
            )
        if "RMS step criterion" in criteria:
            kwargs["convergence_drms"] = P["RMS step criterion"].m_as("Å")
        if "Maximum atomic step criterion" in criteria:
            kwargs["convergence_dmax"] = P["Maximum atomic step criterion"].m_as("Å")

        if convergence_formula == "MolPro":
            kwargs["molcnv"] = True
        elif convergence_formula == "QChem":
            kwargs["qccnv"] = True
        elif convergence_formula == "MOPAC":
            kwargs["convergence_grms"] *= math.sqrt(3 * n_atoms)

        converged = True
        tic = time.perf_counter_ns()
        exception = None
        if P["target"] == "Transition state" and P["calculate hessian"] == "never":
            P["calculate hessian"] = "first"
        with cd(self._working_directory):
            try:
                m = geometric.optimize.run_optimizer(
                    logIni=str(logIni),
                    customengine=customengine,
                    input="optimization.txt",
                    qdata=True,
                    **kwargs,
                )
            except geometric.errors.GeomOptNotConvergedError:
                converged = False
            except Exception as exception:  # noqa: F841
                print(exception)
                converged = False

        toc = time.perf_counter_ns()
        self._results["t_elapsed"] = round((toc - tic) * 1.0e-9, 3)
        self._results["converged"] = converged

        # Make the geometric output readable by removing the ANSI escape sequences
        if logPath.exists():
            text = logPath.read_text()
            text = ansi_escape.sub("", text)
            logPath.write_text(text)

        # Get the optimized energy & geometry
        self._results["energy"] = (
            m.qm_energies[-1] * Q_(1.0, "E_h").to("kJ/mol").magnitude
        )
        coordinates = m.xyzs[-1].reshape(-1, 3)
        gradients = m.qm_grads[-1].reshape(-1, 3)
        self._results["maximum_gradient"] = np.max(np.linalg.norm(gradients, axis=1))
        self._results["rms_gradient"] = np.sqrt(
            np.mean(np.linalg.norm(gradients, axis=1) ** 2)
        )
        self._results["nsteps"] = len(m.qm_energies) - 1
        step = coordinates - m.xyzs[-2].reshape(-1, 3)
        self._results["maximum_step"] = np.max(np.linalg.norm(step, axis=1))

        self.analyze()

        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug("optimized coordinates")
            for i in range(n_atoms):
                logger.debug(
                    f"   {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f} "
                    f"{coordinates[i][2]:8.3f}"
                )

        # Set the coordinates in the configuration
        self._working_configuration.atoms.set_coordinates(
            coordinates, fractionals=False
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug("step optimized coordinates")
            coordinates = self._working_configuration.coordinates
            symbols = self._working_configuration.atoms.symbols
            for i in range(n_atoms):
                logger.debug(
                    f"   {symbols[i]} {coordinates[i][0]:8.3f} "
                    f"{coordinates[i][1]:8.3f} {coordinates[i][2]:8.3f}"
                )
            logger.debug(self._working_configuration.bonds)

        # Clean up the subdirectories
        wd = self._working_directory
        if exception is not None or not converged:
            keep = P["on error"]
            if keep == "delete all subdirectories":
                subdirectories = wd.glob("step_*")
                for subdirectory in subdirectories:
                    shutil.rmtree(subdirectory)
            elif keep == "keep last subdirectory":
                subdirectories = wd.glob("step_*")
                subdirectories = sorted(subdirectories)
                for subdirectory in subdirectories[:-1]:
                    shutil.rmtree(subdirectory)
            if not converged:
                raise RuntimeError(
                    f"Optimization did not converge in {max_steps} steps"
                )
            raise exception from None
        else:
            keep = P["on success"]
            if keep == "delete all subdirectories":
                subdirectories = wd.glob("step_*")
                for subdirectory in subdirectories:
                    shutil.rmtree(subdirectory)
            elif keep == "keep last subdirectory":
                subdirectories = wd.glob("step_*")
                subdirectories = sorted(subdirectories)
                for subdirectory in subdirectories[:-1]:
                    shutil.rmtree(subdirectory)

        # Citation!
        self.read_bibliography()
        self.references.cite(
            raw=self._bibliography["geomeTRIC"],
            alias="geomeTRIC",
            module="seamm_geometric",
            level=1,
            note="Main reference for geomeTRIC.",
        )

        if "seamm_geometric" in self._bibliography:
            try:
                template = string.Template(self._bibliography["seamm_geometric"])

                if "untagged" in __version__ or "unknown" in __version__:
                    # Development version
                    year = datetime.now().year
                    month = datetime.now().month
                else:
                    year, month = __version__.split(".")[0:2]
                try:
                    month = calendar.month_abbr[int(month)].lower()
                except Exception:
                    year = datetime.now().year
                    month = datetime.now().month
                    month = calendar.month_abbr[int(month)].lower()

                citation = template.substitute(
                    month=month, version=__version__, year=str(year)
                )

                title = "seamm_geometric".split("_")
                title = " ".join([s.capitalize() for s in title[0:-2]])
                self.references.cite(
                    raw=citation,
                    alias="seamm_geometric",
                    module="seamm_geometric",
                    level=self.citation_level,
                    note=(f"The principle citation for the {title} step in " "SEAMM."),
                )

            except Exception as e:
                printer.important(f"Exception in citation {type(e)}: {e}")
                printer.important(traceback.format_exc())
