=======
History
=======
2026.11.25 -- Bugfix: the final forces were reported in the wrong units
    * At the end of an optimization the maximum and RMS forces were reported in
      geomeTRIC's units, E_h/bohr, but labelled kJ/mol/Å, so they were about 4961 times
      too small (e.g. "Max Force 0.000016 kJ/mol/Å" for a converged PM7 optimization).
      They are now converted to kJ/mol/Å, as the forces for each step already were.

2026.11.24.1 -- Bugfix: gradient criteria in geomeTRIC's units
    * The RMS and maximum gradient criteria were passed to geomeTRIC in E_h/Å, but
      geomeTRIC reads them in E_h/bohr, so a criterion given in other units, such as a
      custom 0.0015 eV/Å, was 1.89 times too loose. They are now converted to E_h/bohr.
      Needs structure-step 2026.11.24.1 or later, whose presets give their gradient
      criteria in E_h/bohr; with older presets geomeTRIC would be 1.89 times too tight.

2026.11.24 -- Bugfix: a custom convergence stopped the optimization before it ran
    * Describing a custom convergence looked up the criteria under names that are not
      parameters, so any optimization with convergence 'custom' failed with a KeyError
      before it started. The criteria are now named as the parameters are, and a custom
      convergence lists its formula's own criteria.

2026.10.19 -- Bugfix: the error that stopped an optimization was hidden
    * When an error stopped the optimization, e.g. an engine that could not start,
      it was reported as an UnboundLocalError, or as not converging, rather than as
      itself. It is now raised as it is.

2025.5.7 -- Enhancement: Gracefully handle convergence failure
    * Return the latest energy and structure when optimization fails to converge,
      rather than forcing an error.
      
2024.10.20 -- Improved citation handling

2024.10.15 -- Added various convergence metrics to results
    * Added maximum force, gradient, etc. to results.
    * Improved control over when to calculate the Hessian matrix.
    * Fixed an issue converting the units of the gradients.
      
2024.8.2.1 -- Calculator required to have 'implemented_properties' and 'nolabel' defined

2024.8.2 -- Initial version


