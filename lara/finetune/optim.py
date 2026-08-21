"""The Muon optimiser, and the one place that explains how to install it.

Not on PyPI under that name: `pip install muon` fetches "Multimodal omics analysis
framework", an unrelated project that owns the name. That package imports cleanly and
then does not provide the symbol, so the failure lands at optimiser construction with a
message about an attribute rather than about the install — which is why this raises with
instructions rather than letting the ImportError through.

The message was written out verbatim at three call sites; the fourth would have drifted.
"""

from __future__ import annotations

INSTALL_HINT = (
    "the Muon optimiser is not installed. It is not on PyPI under that name -- "
    "`pip install muon` fetches an unrelated omics package that will import but "
    "not provide SingleDeviceMuonWithAuxAdam. Install from source:\n"
    "    pip install git+https://github.com/KellerJordan/Muon"
)


def load_muon():
    """`SingleDeviceMuonWithAuxAdam`, or an ImportError that says how to get it."""
    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return SingleDeviceMuonWithAuxAdam
