"""FastAPI application factory and console entry point for the research cockpit.

This module is the ONLY place in :mod:`futures_engine.ui` that imports the web
stack. The web deps ship in the optional ``ui`` extra (UI-G2); if they are not
installed, importing this module raises a clear :class:`ImportError` naming the
extra to install. ``import futures_engine`` never triggers this path.

The server binds ``127.0.0.1`` only (UI-G4) and HTMX is served from the local
``/static`` mount (UI-G3) -- never a CDN.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # pragma: no cover - exercised via the lazy-import boundary test
    from fastapi import FastAPI, Form, Request
    from fastapi.responses import HTMLResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import ValidationError
except ImportError as exc:  # pragma: no cover - trivial re-raise
    raise ImportError(
        "The research cockpit requires the optional 'ui' extra. "
        'Install it with:  pip install -e ".[ui]"'
    ) from exc

from futures_engine.ui.config_form import (
    INSTRUMENTS,
    config_to_yaml,
    form_to_config,
    prop_preset_choices,
    signal_choices,
)
from futures_engine.ui.data_panel import (
    databento_enabled,
    generate_synthetic,
    list_snapshots,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _PACKAGE_ROOT / "static"
_TEMPLATES_DIR = _PACKAGE_ROOT / "templates"

HOST = "127.0.0.1"
PORT = 8000

# Where snapshots live on disk. Overridable via env so tests (and operators) can
# point the cockpit at an isolated store without touching the repo default.
_DEFAULT_SNAPSHOT_ROOT = "data/snapshots"
# Where the Configure screen writes each run's config.yaml (the U5 handoff root).
_DEFAULT_RUNS_ROOT = "runs"
# Directory holding instruments/prop_rules YAML (the repo configs).
_DEFAULT_CONFIGS_DIR = "configs"

# Field-name -> label for turning a pydantic ValidationError into a readable line.
_GATE_DEFAULTS = {"gate_min_dsr_p": "0.95", "gate_max_pbo": "0.5", "gate_min_survival": "0.90"}
_FORM_DEFAULTS = {
    "run_id": "",
    "instrument": "MES",
    "signal": "donchian_breakout",
    "grid": "window: 5, 10, 15, 20, 30, 40, 55, 100",
    "snapshot_hash": "",
    "prop_preset": "topstep_50k",
    "seed": "7",
    "n_splits": "5",
    "embargo_frac": "0.02",
    "pbo_partitions": "8",
    "bootstrap_n": "500",
    "survival_n_paths": "300",
    "survival_horizon_days": "30",
    "survival_contracts": "1",
    **_GATE_DEFAULTS,
}


def _validation_error_lines(exc: ValidationError) -> list[str]:
    """Render a pydantic ValidationError as human-readable ``loc: message`` lines."""
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"{loc}: {err['msg']}")
    return lines


def create_app(
    *,
    snapshot_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    configs_dir: str | Path | None = None,
) -> FastAPI:
    """Build and return the cockpit FastAPI app.

    Mounts ``/static`` (vendored HTMX + cockpit CSS), wires the Jinja template
    environment, and serves the nav shell plus the Data screen. The UI stays
    thin (UI-G1): routes only orchestrate the store + shared generator.

    ``snapshot_root`` selects the snapshot store; it defaults to
    ``$FE_SNAPSHOT_ROOT`` or ``data/snapshots``.
    """
    root = Path(
        snapshot_root
        if snapshot_root is not None
        else os.environ.get("FE_SNAPSHOT_ROOT", _DEFAULT_SNAPSHOT_ROOT)
    )
    runs = Path(
        runs_root if runs_root is not None else os.environ.get("FE_RUNS_ROOT", _DEFAULT_RUNS_ROOT)
    )
    configs = Path(
        configs_dir
        if configs_dir is not None
        else os.environ.get("FE_CONFIGS_DIR", _DEFAULT_CONFIGS_DIR)
    )
    app = FastAPI(title="futures-engine research cockpit")
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html")

    @app.get("/data", response_class=HTMLResponse)
    def data(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "data.html",
            {
                "snapshots": list_snapshots(root),
                "databento_enabled": databento_enabled(),
            },
        )

    @app.post("/data/synthetic")
    def create_synthetic(
        symbol_root: str = Form("MES"),
        n_bars: int = Form(2500),
        seed: int = Form(7),
    ) -> RedirectResponse:
        generate_synthetic(root, symbol_root=symbol_root, n_bars=n_bars, seed=seed)
        # POST/redirect/GET: refresh the list without re-submitting on reload.
        return RedirectResponse(url="/data", status_code=303)

    def _configure_context(values: dict[str, str], errors: list[str] | None) -> dict[str, object]:
        return {
            "values": values,
            "errors": errors,
            "signals": signal_choices(),
            "presets": prop_preset_choices(configs),
            "instruments": list(INSTRUMENTS),
            "snapshots": list_snapshots(root),
        }

    @app.get("/configure", response_class=HTMLResponse)
    def configure(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "configure.html", _configure_context(dict(_FORM_DEFAULTS), None)
        )

    @app.post("/configure")
    async def configure_submit(request: Request) -> Response:
        form = {key: str(value) for key, value in (await request.form()).items()}
        values = {**_FORM_DEFAULTS, **form}
        try:
            cfg = form_to_config(form)
        except ValidationError as exc:
            context = _configure_context(values, _validation_error_lines(exc))
            return templates.TemplateResponse(request, "configure.html", context, status_code=200)
        except ValueError as exc:
            context = _configure_context(values, [str(exc)])
            return templates.TemplateResponse(request, "configure.html", context, status_code=200)

        # Handoff to U5: write the validated config to a run-scoped path and redirect
        # to the Run screen. Contract: runs_root/<run_id>/config.yaml holds the config;
        # U5's /runs/<run_id> loads it via PipelineConfig.load and executes the pipeline.
        config_to_yaml(cfg, runs / cfg.run_id / "config.yaml")
        return RedirectResponse(url=f"/runs/{cfg.run_id}", status_code=303)

    return app


def main() -> None:
    """Console entry point (``futures-cockpit``): run the local-only server."""
    import uvicorn

    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
