"""HTTP surface, one module per area of the app.

Split out of a single 1,900-line ``app.py``. Each module owns an ``APIRouter`` and the
request models for its own endpoints; ``lara.serve.app`` builds the application and
includes them. Shared state plumbing lives in :mod:`lara.serve.deps`.

Order matters when routes could shadow each other -- see the note on ``/api/meminfo``
in :mod:`lara.serve.routes.health`.
"""

from lara.serve.routes import (
    dataset,
    health,
    library,
    models,
    papers,
    research,
    retrieval,
    taste,
    ui,
)

#: Included in this order by lara.serve.app.
ROUTERS = [
    ui.router,
    health.router,
    papers.router,
    retrieval.router,
    library.router,
    taste.router,
    models.router,
    dataset.router,
    research.router,
]

__all__ = ["ROUTERS"]
