"""Reproduction of Bent et al., npj Digital Medicine (2021) 4:89.

Engineering digital biomarkers of interstitial glucose from noninvasive smartwatches.
https://doi.org/10.1038/s41746-021-00465-w

This package follows the paper and Supplementary Information as the primary
authority. Implementation choices that are not stated in those sources are
recorded in UNCERTAINTIES (see `uncertainties.py` and the validation report).
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
