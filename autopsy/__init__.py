"""ActarusLab · MODEL AUTOPSY — forensic validation for QSAR models."""
from .engine import run_autopsy, AutopsyError, AutopsyResult

__all__ = ["run_autopsy", "AutopsyError", "AutopsyResult"]
__version__ = "0.1.0"
