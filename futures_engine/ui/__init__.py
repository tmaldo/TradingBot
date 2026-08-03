"""futures_engine.ui: local research cockpit (FastAPI + HTMX).

A THIN orchestration layer over the audited pipeline (UI-G1). Importing this
package is deliberately cheap: it pulls in **no** web dependencies. The heavy
imports (``fastapi``, ``uvicorn``, ``jinja2``) live inside
:mod:`futures_engine.ui.app` and are only required when you actually build or
run the app (UI-G2). This keeps ``import futures_engine`` and the core test
suite working with none of the optional extras installed.
"""

from __future__ import annotations
