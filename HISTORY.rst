=======
History
=======
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


