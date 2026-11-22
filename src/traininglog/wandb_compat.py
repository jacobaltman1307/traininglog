"""Thin Weights & Biases compatibility shim backed by TrainingLog."""

from __future__ import annotations

from typing import Any

from .sdk.run import Run, init as tl_init

run: Run | None = None
config: Any = None
summary: Any = None


class Table:
    """Minimal wandb.Table replacement."""

    def __init__(self, data: list[list[Any]] | None = None, columns: list[str] | None = None):
        self.data = list(data or [])
        self.columns = list(columns or [])

    def to_markdown(self) -> str:
        if not self.columns:
            return ""
        header = "| " + " | ".join(str(c) for c in self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        rows = [
            "| " + " | ".join(str(cell) for cell in row) + " |"
            for row in self.data
        ]
        return "\n".join([header, sep, *rows])


class _BarChart:
    def __init__(self, table: Table, label: str, value: str, title: str = ""):
        self.table = table
        self.label = label
        self.value = value
        self.title = title


class _PlotNamespace:
    @staticmethod
    def bar(table: Table, label: str, value: str, title: str = "") -> _BarChart:
        return _BarChart(table=table, label=label, value=value, title=title)


plot = _PlotNamespace()


def init(
    *,
    project: str = "default",
    entity: str | None = None,
    sync_tensorboard: bool = False,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    notes: str = "",
    mode: str | None = None,
    id: str | None = None,
    dir: str | None = None,
    **kwargs: Any,
) -> Run:
    """Drop-in replacement for wandb.init(...)."""
    global run, summary
    globals()["config"]

    resolved_tags = list(tags or [])
    if entity:
        resolved_tags.append(f"entity:{entity}")

    run = tl_init(
        project=project,
        name=name,
        run_id=id,
        group=group,
        tags=resolved_tags,
        notes=notes,
        config=config,
        mode=mode,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "arch",
                "obs_dim",
                "manifest_hash",
                "parent_checkpoint",
                "total_timesteps",
                "server_url",
                "api_token",
                "capture_console",
                "capture_system",
                "spool_dir",
                "transport",
                "finish_timeout",
            )
        },
    )
    globals()["config"] = run.config
    summary = run.summary
    return run


def log(
    data: dict[str, Any],
    step: int | None = None,
    commit: bool = True,
) -> None:
    """Drop-in replacement for wandb.log(...)."""
    if run is None:
        raise RuntimeError("Call wandb.init() before wandb.log().")
    resolved_step = step
    if resolved_step is None and "global_step" in data:
        try:
            resolved_step = int(data["global_step"])
        except (TypeError, ValueError):
            resolved_step = None
    elif resolved_step is None and "step" in data:
        try:
            resolved_step = int(data["step"])
        except (TypeError, ValueError):
            resolved_step = None

    scalars: dict[str, Any] = {}
    for k, v in data.items():
        if k in ("global_step", "step") and resolved_step is not None:
            continue
        if isinstance(v, _BarChart):
            title = v.title or k
            run.summary[k] = {
                "title": title,
                "columns": v.table.columns,
                "rows": v.table.data,
            }
            run.log_text(k, f"{title}\n{v.table.to_markdown()}", step=resolved_step)
        elif isinstance(v, Table):
            run.summary[k] = {"columns": v.columns, "rows": v.data}
            run.log_text(k, v.to_markdown(), step=resolved_step)
        else:
            scalars[k] = v

    if scalars:
        run.log(scalars, step=resolved_step)


def finish(exit_code: int = 0) -> None:
    """Finish the active wandb_compat run."""
    global run, config, summary
    if run is not None:
        run.finish(exit_code=exit_code)
        run = None
        config = None
        summary = None
