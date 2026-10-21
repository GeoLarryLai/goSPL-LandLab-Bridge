"""Quiet Model load + ASCII tqdm helpers (black/default terminal bars, not green)."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union


def _tqdm_kwargs(desc: str, total: Optional[int] = None) -> dict:
    # ascii=True → "#"-style bar; leave colour unset / None so terminals stay plain
    return dict(
        desc=desc,
        total=total,
        ascii=True,
        leave=True,
        file=sys.stderr,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


@contextlib.contextmanager
def silence_stdout(enabled: bool = True):
    """Swallow stdout (goSPL banners) while keeping stderr free for tqdm."""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def landlab_substeps_for_dt(dt_yr: float, *, target_sub_yr: float = 10_000.0) -> int:
    """
    Choose BridgeRunner ``substeps`` so each Landlab inner step ≈ ``target_sub_yr``.

    Example: ``dt=1e5``, ``target_sub_yr=1e4`` → 10 substeps.
    """
    dt = float(dt_yr)
    sub = float(target_sub_yr)
    if dt <= 0 or sub <= 0:
        return 1
    return max(1, int(round(dt / sub)))


def _yml_with_timing(
    yml: Path,
    *,
    dt: Optional[float] = None,
    end: Optional[float] = None,
    tout: Optional[float] = None,
    strat: Optional[float] = None,
) -> Path:
    """Write a temp YAML with overridden ``time`` (and tectonics end if ``end`` set)."""
    import yaml

    with open(yml) as f:
        cfg = yaml.safe_load(f)
    time = dict(cfg.get("time") or {})
    if end is not None:
        time["end"] = float(end)
    if dt is not None:
        time["dt"] = float(dt)
        # Keep output + strata locked to dt unless caller overrides
        if tout is None:
            time["tout"] = float(dt)
        if strat is None:
            time["strat"] = float(dt)
    if tout is not None:
        time["tout"] = float(tout)
    if strat is not None:
        time["strat"] = float(strat)
    cfg["time"] = time
    if end is not None and cfg.get("tectonics"):
        for ev in cfg["tectonics"]:
            if isinstance(ev, dict) and "end" in ev:
                ev["end"] = float(end)

    tmp_dir = Path(tempfile.mkdtemp(prefix="gospl_landlab_yml_"))
    out = tmp_dir / yml.name
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out


def _prepare_output_dir_from_yml(yml: Path) -> None:
    """
    When YAML has ``makedir: False``, goSPL deletes then recreates ``output.dir``.
    On cloud-synced folders ``shutil.rmtree(..., ignore_errors=True)`` often
    leaves the directory behind → ``FileExistsError`` on ``os.makedirs``.

    Clear (or rename aside) the output path here before ``Model()`` so re-runs work.
    """
    import shutil
    import time

    import yaml

    with open(yml) as f:
        cfg = yaml.safe_load(f) or {}
    out_cfg = cfg.get("output") or {}
    rel = out_cfg.get("dir")
    if not rel:
        return
    # goSPL only overwrites when makedir is False (True → numbered ``dir_N``)
    if bool(out_cfg.get("makedir", False)):
        return

    path = (Path.cwd() / rel).resolve()
    if not path.exists():
        return

    for _ in range(6):
        try:
            shutil.rmtree(path)
        except OSError:
            pass
        if not path.exists():
            return
        time.sleep(0.15)

    # Last resort: move aside so goSPL can create a fresh folder
    aside = path.with_name(f"{path.name}_old_{int(time.time())}")
    try:
        path.rename(aside)
    except OSError as exc:
        raise FileExistsError(
            f"Could not clear goSPL output dir {path} (cloud-sync lock?). "
            f"Delete it manually and re-run."
        ) from exc


def _restore_zero_clinslp(model, yml: Path) -> None:
    """
    goSPL's inputparser does ``clinSlp = max(1e-6, clinSlp)``, which forces the
    VTK coastline path in ``seaChange`` even when YAML says ``clinSlp: 0``.

    If the YAML requested 0, put it back so marine deposition still runs
    (``seadepo: True``) without VTK contouring — shelf cap = flat sealevel.
    """
    import yaml

    with open(yml) as f:
        cfg = yaml.safe_load(f) or {}
    requested = (cfg.get("diffusion") or {}).get("clinSlp", None)
    if requested is None:
        return
    if float(requested) == 0.0:
        model.clinSlp = 0.0


def load_model(
    yml: Union[str, Path],
    *,
    verbose: bool = False,
    dt: Optional[float] = None,
    end: Optional[float] = None,
    tout: Optional[float] = None,
    strat: Optional[float] = None,
):
    """
    Build ``gospl.model.Model``.

    ``verbose=True``: stock goSPL text.
    ``verbose=False`` (default): silence init banners; show a one-step ASCII tqdm.

    Optional ``dt`` / ``end`` / ``tout`` / ``strat`` override the YAML ``time``
    block via a temp file (so strata layer counts match the chosen step).
    When ``dt`` is set and ``tout``/``strat`` are omitted, both default to ``dt``.
    """
    from gospl.model import Model
    from tqdm.auto import tqdm

    from .gospl_patches import apply_gospl_patches

    apply_gospl_patches()

    yml_path = Path(yml)
    if any(v is not None for v in (dt, end, tout, strat)):
        yml_path = _yml_with_timing(
            yml_path, dt=dt, end=end, tout=tout, strat=strat
        )
    _prepare_output_dir_from_yml(yml_path)
    yml_s = str(yml_path)

    if verbose:
        m = Model(yml_s, verbose=True)
    else:
        with silence_stdout(True):
            with tqdm(**_tqdm_kwargs("Loading goSPL model", total=1)) as pbar:
                m = Model(yml_s, verbose=False)
                pbar.update(1)
    _restore_zero_clinslp(m, yml_path)
    return m


def estimate_n_steps(model) -> int:
    """Approximate number of outer time-loop iterations left (inclusive of end dump)."""
    t0 = float(getattr(model, "tNow", 0.0))
    t1 = float(model.tEnd)
    dt = float(model.dt)
    if dt <= 0:
        return 1
    # BridgeRunner / runProcesses: vis at tNow, then advance until tNow==tEnd break
    return max(1, int(round((t1 - t0) / dt)) + 1)


def format_run_done(model, *, label: str, out_dir: Optional[Union[str, Path]] = None) -> str:
    """Short informative one-liner after a finished run."""
    end_myr = float(model.tEnd) / 1e6
    dt_kyr = float(model.dt) / 1e3
    n = max(1, int(round(float(model.tEnd) / float(model.dt))))
    out = out_dir
    if out is None:
        out = getattr(model, "outputDir", None) or getattr(model, "output", None)
    out_s = str(out) if out else "?"
    return f"{label}: {end_myr:g} Myr · {n}×{dt_kyr:g} kyr steps · wrote {out_s}"


def destroy_model(model, *, verbose: bool = False) -> None:
    """Call ``model.destroy()``; silence goSPL shutdown banners when quiet."""
    with silence_stdout(enabled=not verbose):
        model.destroy()
