"""
wandb_utils.py — thin helpers around Weights & Biases for experiment tracking.

The competition rules forbid editing ``model.py`` and ``evaluate.py`` but allow
us to instrument ``solution.py``, ``probe.py``, ``aggregation.py``, and
``splitting.py``. This module centralises the wandb integration so the
student-editable files stay readable.

Behaviour
---------
* Always-on by default. Set ``WANDB_MODE=disabled`` (or ``WANDB_DISABLED=true``)
  to make every wandb call a no-op without touching the surrounding code.
* If the ``wandb`` package is not importable, every helper degrades to a no-op
  and the rest of the pipeline still runs.
* All public helpers are safe to call even before ``init_run`` has been called.

Environment variables consumed
------------------------------
``WANDB_API_KEY``  -- API key for the wandb account that should own the run
``WANDB_PROJECT``  -- project name (default ``smiles-2026-hallucination``)
``WANDB_ENTITY``   -- team / username (optional; the user/team the run is filed under)
``WANDB_RUN_NAME`` -- pretty run name (optional)
``WANDB_TAGS``     -- comma-separated tags (optional)
``WANDB_NOTES``    -- free-text run description (optional)
``WANDB_MODE``     -- ``online`` | ``offline`` | ``disabled`` (default online)
``WANDB_DISABLED`` -- legacy switch; ``true`` forces ``mode=disabled``

Local credentials override (``.env.local``)
-------------------------------------------
On import, this module loads ``.env`` and then ``.env.local`` from the repo
root if they exist. Values in those files override values inherited from the
shell environment, which is the right behaviour when the host already has a
team-wide ``WANDB_API_KEY`` exported and the developer wants their personal
account to be used instead. ``.env.local`` is git-ignored; see
``.env.example`` for the template.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover -- offline / not installed
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


_DEFAULT_PROJECT = "smiles-2026-hallucination"
_REPO_ROOT = Path(__file__).resolve().parent
_DOTENV_LOADED = False


def _load_dotenv_files() -> None:
    """Load ``.env`` then ``.env.local`` from the repo root.

    Values found in these files override any pre-existing values in
    ``os.environ``. This is deliberate: typical use case is a host with a
    team-wide ``WANDB_API_KEY`` already exported, and a developer who wants
    their personal account to be used for this project.

    No third-party dependency: a tiny ``KEY=VALUE`` parser is sufficient.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    for filename in (".env", ".env.local"):
        path = _REPO_ROOT / filename
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip optional matching quotes around the value.
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if not key:
                    continue
                os.environ[key] = value
            print(f"[wandb] loaded credentials from {path.name}")
        except OSError as exc:  # pragma: no cover -- unreadable file
            print(f"[wandb] skipped {path.name}: {exc}")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_mode() -> str:
    """Honour ``WANDB_DISABLED`` as a legacy shortcut for ``WANDB_MODE=disabled``."""
    if _env_bool("WANDB_DISABLED"):
        return "disabled"
    return os.environ.get("WANDB_MODE", "online")


def _parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def is_active() -> bool:
    """Return ``True`` if a live wandb run is currently attached to the process."""
    return _WANDB_AVAILABLE and getattr(wandb, "run", None) is not None


def init_run(
    config: Mapping[str, Any] | None = None,
    *,
    project: str | None = None,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    notes: str | None = None,
    job_type: str = "probe-train",
) -> Any:
    """Initialise a wandb run if wandb is available and not disabled.

    Args:
        config:   Dict of hyperparameters / settings to record under
                  ``wandb.config``. Captures everything you would want to
                  filter runs by in the dashboard.
        project:  Override the project name (defaults to env var or
                  ``smiles-2026-hallucination``).
        name:     Pretty name for this run. Defaults to ``WANDB_RUN_NAME`` env
                  var or wandb's auto-generated name.
        tags:     Optional sequence of tags. Merged with ``WANDB_TAGS`` env var.
        notes:    Free-text description. Defaults to ``WANDB_NOTES`` env var.
        job_type: Logical phase of the run. Useful when grouping extraction-only
                  vs probe-only runs later.

    Returns:
        The wandb run object, or ``None`` if wandb is unavailable / disabled.
    """
    _load_dotenv_files()

    if not _WANDB_AVAILABLE:
        print("[wandb] package not installed -- tracking disabled.")
        return None

    mode = _resolve_mode()
    if mode == "disabled":
        print("[wandb] mode=disabled -- tracking turned off via env var.")

    env_tags = _parse_tags(os.environ.get("WANDB_TAGS"))
    merged_tags = list(env_tags) + list(tags or [])
    if not merged_tags:
        merged_tags = None  # type: ignore[assignment]

    run = wandb.init(
        project=project or os.environ.get("WANDB_PROJECT", _DEFAULT_PROJECT),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=name or os.environ.get("WANDB_RUN_NAME") or None,
        tags=merged_tags,
        notes=notes or os.environ.get("WANDB_NOTES") or None,
        job_type=job_type,
        mode=mode,
        config=dict(config) if config else None,
        reinit=True,
    )

    if run is not None and mode != "disabled":
        entity = run.entity or "<default-entity>"
        print(f"[wandb] run started: {run.name}  entity={entity}  ({run.url})")
    return run


def log(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    """Log a dict of metrics to the active run. No-op when wandb is inactive."""
    if not is_active():
        return
    wandb.log(dict(metrics), step=step)


def update_summary(metrics: Mapping[str, Any]) -> None:
    """Write final / averaged metrics to ``wandb.run.summary``.

    Use this for values that should appear as columns on the wandb runs table
    (e.g. ``avg_test_accuracy``, ``avg_test_auroc``).
    """
    if not is_active():
        return
    for key, value in metrics.items():
        wandb.run.summary[key] = value  # type: ignore[union-attr]


def log_fold_table(fold_results: Sequence[Mapping[str, Any]]) -> None:
    """Log a wandb.Table with one row per fold so per-fold metrics are filterable."""
    if not is_active() or not fold_results:
        return
    columns = sorted({k for r in fold_results for k in r.keys()})
    table = wandb.Table(columns=columns)  # type: ignore[union-attr]
    for row in fold_results:
        table.add_data(*[row.get(c) for c in columns])
    wandb.log({"fold_results": table})


def finish_run(exit_code: int | None = None) -> None:
    """Close the active wandb run. Safe to call multiple times."""
    if not is_active():
        return
    wandb.finish(exit_code=exit_code)
