"""BridgeRunner: wrap gospl.Model timesteps with optional Landlab process slots."""

from __future__ import annotations

from time import process_time
from typing import Any, Optional

import numpy as np

from .capability import probe_component
from .config import BridgeConfig, SlotSpec
from .flow_adapter import map_gospl_flow_to_landlab, run_landlab_flow
from .loader import instantiate_component
from .mirror import (
    MirrorState,
    build_irregular_mirror,
    build_pull_weight_cache,
    build_push_weight_cache,
    build_raster_mirror,
    pull_elevation,
    pull_field_to_gospl,
    pull_weights_ready,
    push_elevation,
    push_weights_ready,
)
from .mpi_io import get_comm, scatter_field, warn_if_large
from .sediment import choose_sediment_volume, extract_landlab_sediment


class BridgeRunner:
    """
    Outer time loop mirroring goSPL ``Model.runProcesses`` for goSPL ≥2026
    (``SPL.erodepSPL`` / ``hillSLP.getHillslope`` / ``GridProcess.applyFlexure``).

    When a slot is ``gospl.*``, the corresponding goSPL method is called.
    When ``landlab.*``, state is mirrored, the component runs on rank 0, and
    elevation (and optional sediment/soil) is written back.
    """

    def __init__(self, model, config: BridgeConfig):
        from .gospl_patches import apply_gospl_patches

        apply_gospl_patches()
        self.model = model
        self.config = config
        self.comm = get_comm()
        self.rank = self.comm.Get_rank()
        self.mirror: Optional[MirrorState] = None
        """Raster mirror (DDTD + landslide / legacy Landlab-fluvial path)."""
        self.mirror_irreg: Optional[MirrorState] = None
        """Unused in demo Run B (kept for optional legacy irregular DDTD)."""
        self._fluvial_comp = None
        self._hillslope_comp = None
        self._landslide_comp = None
        self._flow_comp = None
        # Reused per-step raster buffers (Δz / Δsoil); allocated on first Landlab step
        self._buf_z_before: Optional[np.ndarray] = None
        self._buf_soil_before: Optional[np.ndarray] = None
        self._buf_dz: Optional[np.ndarray] = None
        self._buf_dsoil: Optional[np.ndarray] = None
        self._buf_pull: Optional[np.ndarray] = None
        self._buf_pull_soil: Optional[np.ndarray] = None
        self._needs_landlab = self._any_landlab_slot()
        if self._needs_landlab:
            self._validate_slots()

    def _any_landlab_slot(self) -> bool:
        for slot in (self.config.fluvial, self.config.hillslope, self.config.mass_wasting):
            if slot and getattr(slot, "is_landlab", False):
                return True
            if slot and slot.backend.startswith("landlab"):
                return True
        return self.config.flow == "landlab"

    def _validate_slots(self) -> None:
        for slot in (self.config.fluvial, self.config.hillslope, self.config.mass_wasting):
            if not slot or getattr(slot, "is_off", False):
                continue
            if not slot.backend.startswith("landlab"):
                continue
            name = slot.backend.split(".", 1)[-1]
            cap = probe_component(name)
            if cap.status == "unsupported":
                raise RuntimeError(f"Landlab component unsupported: {cap.notes}")
            if cap.status == "needs_landlab_flow" and self.config.flow != "landlab":
                if self.rank == 0 and bool(getattr(self.model, "verbose", False)):
                    print(
                        f"[gospl_landlab] NOTE: {name} uses goSPL-mapped flow "
                        f"({cap.notes}).",
                        flush=True,
                    )

    def _landlab_stages_before_sed(self) -> list:
        """Landlab stages that run after goSPL FA (and after goSPL SPL if fluvial is gospl)."""
        stages = []
        if self.config.fluvial.is_landlab:
            stages.append("fluvial")
        if self.config.hillslope.is_landlab:
            stages.append("hillslope")
        if self.config.mass_wasting and self.config.mass_wasting.is_landlab:
            stages.append("mass_wasting")
        return stages

    def _landlab_hillslope_mw_stages(self) -> list:
        """Landlab hillslope / mass-wasting only (for goSPL-SPL + Landlab HyLands demos)."""
        stages = []
        if self.config.hillslope.is_landlab:
            stages.append("hillslope")
        if self.config.mass_wasting and self.config.mass_wasting.is_landlab:
            stages.append("mass_wasting")
        return stages

    # ------------------------------------------------------------------ API
    def run(self) -> None:
        m = self.model
        # Mixin classes as in goSPL Model.runProcesses (2026+)
        from gospl.eroder import SPL as _SPL
        from gospl.flow import FAMesh as _FAMesh
        from gospl.mesher import Tectonics as _Tectonics
        from gospl.mesher import UnstMesh as _UnstMesh
        from gospl.sed import SEAMesh as _SEAMesh
        from gospl.sed import SEDMesh as _SEDMesh
        from gospl.sed import STRAMesh as _STRAMesh
        from gospl.sed import hillSLP as _hillSLP
        from gospl.tools import GridProcess as _GridProcess
        from gospl.tools import WriteMesh as _WriteMesh

        from .progress import _tqdm_kwargs, estimate_n_steps, silence_stdout

        verbose = bool(getattr(m, "verbose", False))
        ll_all = self._landlab_stages_before_sed()
        ll_hs_mw = self._landlab_hillslope_mw_stages()

        # Build Landlab mirrors + remap weight tables BEFORE the time-step bar
        if self._needs_landlab or ll_hs_mw or ll_all:
            with silence_stdout(enabled=not verbose):
                self._ensure_mirrors()

        use_tqdm = (not verbose) and self.rank == 0
        pbar = None
        if use_tqdm:
            from tqdm.auto import tqdm

            pbar = tqdm(**_tqdm_kwargs("Time steps", total=estimate_n_steps(m)))

        # Quiet mode: hide goSPL/Landlab chatter; tqdm lives on stderr
        with silence_stdout(enabled=not verbose):
            while m.tNow <= m.tEnd:
                tstep = process_time()

                if getattr(m, "flexOn", False):
                    m.flexCount = getattr(m, "flexCount", 0) + 1

                _Tectonics.updatePaleoZ(m)
                _WriteMesh.visModel(m)
                if m.tNow == m.tEnd:
                    if pbar is not None:
                        pbar.update(1)
                    break

                new_layer = m.tNow >= m.saveStrat
                if new_layer:
                    m.stratStep += 1
                    m.saveStrat += m.strat

                _Tectonics.getTectonics(m)

                if not m.fast:
                    # FA #1 — before fluvial
                    _FAMesh.flowAccumulation(m)

                    if self.config.fluvial.is_gospl:
                        _SPL.erodepSPL(m)
                        # Raster DDTD (+ landslider); Δz + Δsoil write-back (no FA between)
                        if ll_hs_mw:
                            self._run_gospl_spl_landlab_hylands()
                    elif self.config.fluvial.is_landlab and ll_all:
                        # Legacy: Landlab fluvial (± hillslope/MW) on one raster
                        self._run_landlab_processes(stages=ll_all)

                    if not m.nodep:
                        # FA #2 before sed: stock post-SPL review, or post-landslide
                        # after Landlab hillslope/MW. Skip when Landlab fluvial owns the step.
                        if self.config.fluvial.is_gospl:
                            _FAMesh.flowAccumulation(m)
                        _SEDMesh.sedChange(m)
                        if m.seaDepo:
                            _SEAMesh.seaChange(m)

                    # goSPL hillslope only when explicitly requested (not off / landlab)
                    if self.config.hillslope.is_gospl:
                        _hillSLP.getHillslope(m)

                    if verbose:
                        if hasattr(_STRAMesh, "logDualState"):
                            _STRAMesh.logDualState(m)
                        if hasattr(_STRAMesh, "logProvState"):
                            _STRAMesh.logProvState(m)

                if new_layer:
                    _STRAMesh.getCompaction(m)

                if m.flexOn:
                    interval = int(getattr(m, "flex_interval", 1) or 1)
                    if m.flexCount % interval == interval - 1:
                        _GridProcess.applyFlexure(m)

                if m.tNow < m.tEnd:
                    _UnstMesh.applyForces(m)

                m.tNow += m.dt

                if pbar is not None:
                    pbar.set_postfix_str(f"t={m.tNow/1e6:.2f} Myr", refresh=False)
                    pbar.update(1)

                if self.rank == 0 and verbose:
                    print(
                        "--- step done in %0.02f s (now %d yr)"
                        % (process_time() - tstep, m.tNow),
                        flush=True,
                    )

        if pbar is not None:
            pbar.close()

    # ----------------------------------------------------------- Landlab path
    def _needs_irregular_mirror(self) -> bool:
        """Irregular Voronoi DDTD path (off for demo; raster DDTD used instead)."""
        return False

    def _needs_raster_mirror(self) -> bool:
        if self.config.fluvial.is_landlab:
            return True
        if self.config.mass_wasting and self.config.mass_wasting.is_landlab:
            return True
        # Demo Run B: Landlab DDTD (+ optional HyLands) on raster after goSPL SPL
        if self.config.hillslope.is_landlab:
            return True
        return False

    def _ensure_mirrors(self) -> None:
        need_irreg = self._needs_irregular_mirror()
        need_raster = self._needs_raster_mirror()
        mirrors_ready = (not need_irreg or self.mirror_irreg is not None) and (
            not need_raster or self.mirror is not None
        )

        verbose = bool(getattr(self.model, "verbose", False))
        if not mirrors_ready:
            x, y, z = self._gather_xyz()
            if self.rank == 0:
                warn_if_large(z.size, self.config.gather_warn_nodes, rank=0)
                scale = float(getattr(self.config, "raster_scale", 1.0) or 1.0)
                # Raster FA / DDTD / landslider stay float64 (Landlab Cython flow directors).
                irreg_dtype = (
                    np.float32
                    if bool(getattr(self.config, "landlab_float32", False))
                    else np.float64
                )
                if need_irreg and irreg_dtype == np.float32:
                    from .landlab_dtype import patch_ddtd_float32

                    patch_ddtd_float32()

                def _build() -> None:
                    if need_irreg and self.mirror_irreg is None:
                        self.mirror_irreg = build_irregular_mirror(
                            x, y, z, dtype=irreg_dtype
                        )
                        self._build_irregular_components()
                    if need_raster and self.mirror is None:
                        # Raster DDTD+FA need stock float64 field dtypes (undo sticky float32 patch).
                        from .landlab_dtype import restore_ddtd_float64

                        restore_ddtd_float64()
                        self.mirror = build_raster_mirror(
                            x, y, z, scale=scale, dtype=np.float64
                        )
                        self._build_raster_components()

                if verbose:
                    _build()
                else:
                    from tqdm.auto import tqdm

                    from .progress import _tqdm_kwargs

                    with tqdm(
                        **_tqdm_kwargs("Landlab mirror + components", total=1)
                    ) as pbar:
                        _build()
                        pbar.update(1)

        # Weight tables: separate tqdm bars, before any "Time steps" progress
        if self.rank == 0 and need_raster and self.mirror is not None:
            if self.mirror.use_interp:
                if not push_weights_ready(self.mirror):
                    build_push_weight_cache(self.mirror, show_progress=True)
                if not pull_weights_ready(self.mirror):
                    build_pull_weight_cache(self.mirror, show_progress=True)

        # Rank 0 may spend a long time on mirror/weights; keep MPI in lockstep
        self.comm.Barrier()

    def _ensure_mirror(self) -> None:
        """Backward-compatible: ensure whatever mirrors the current path needs."""
        self._ensure_mirrors()

    def _seed_soil_fields(self, g, dtype=None) -> None:
        kw = {}
        if dtype is not None:
            kw["dtype"] = np.dtype(dtype)
        if "soil__depth" not in g.at_node:
            g.add_zeros("soil__depth", at="node", **kw)
        if "bedrock__elevation" not in g.at_node:
            br = g.add_zeros("bedrock__elevation", at="node", **kw)
            br[:] = g.at_node["topographic__elevation"] - g.at_node["soil__depth"]
        if "soil_production__rate" not in g.at_node:
            g.add_zeros("soil_production__rate", at="node", **kw)

    def _build_irregular_components(self) -> None:
        assert self.mirror_irreg is not None
        g = self.mirror_irreg.grid
        self._seed_soil_fields(g, dtype=self.mirror_irreg.dtype)
        if self.config.hillslope.is_landlab and "DepthDependentTaylor" in self.config.hillslope.backend:
            soil = g.at_node["soil__depth"]
            if float(np.max(soil)) <= 0.0:
                soil[:] = 1.0
                g.at_node["bedrock__elevation"][:] = (
                    g.at_node["topographic__elevation"] - soil
                )
        if self.config.hillslope.is_landlab:
            name = self.config.hillslope.backend.split(".", 1)[-1]
            self._hillslope_comp = instantiate_component(
                g, name, self.config.hillslope.kwargs
            )

    def _build_raster_components(self) -> None:
        assert self.mirror is not None
        g = self.mirror.grid
        self._seed_soil_fields(g, dtype=self.mirror.dtype)

        # DDTD (and other Landlab hillslope) on the raster for demo + legacy paths
        if self.config.hillslope.is_landlab and self._hillslope_comp is None:
            if "DepthDependentTaylor" in self.config.hillslope.backend:
                soil = g.at_node["soil__depth"]
                if float(np.max(soil)) <= 0.0:
                    soil[:] = 1.0
                    g.at_node["bedrock__elevation"][:] = (
                        g.at_node["topographic__elevation"] - soil
                    )
            name = self.config.hillslope.backend.split(".", 1)[-1]
            self._hillslope_comp = instantiate_component(
                g, name, self.config.hillslope.kwargs
            )

        from .flow_adapter import ensure_flow_fields, prepare_hylands_routing

        ensure_flow_fields(g)

        if self.config.flow == "landlab" and self._flow_comp is None:
            self._flow_comp = instantiate_component(
                g,
                self.config.landlab_flow_component,
                self.config.landlab_flow_kwargs,
            )
            self._call_comp(self._flow_comp, 1.0)

        if self.config.fluvial.is_landlab and self._fluvial_comp is None:
            name = self.config.fluvial.backend.split(".", 1)[-1]
            self._fluvial_comp = instantiate_component(
                g, name, self.config.fluvial.kwargs
            )

        if (
            self.config.mass_wasting
            and not getattr(self.config.mass_wasting, "is_off", False)
            and self.config.mass_wasting.backend.startswith("landlab")
            and self._landslide_comp is None
        ):
            if "BedrockLandslider" in self.config.mass_wasting.backend:
                prepare_hylands_routing(g)
            name = self.config.mass_wasting.backend.split(".", 1)[-1]
            self._landslide_comp = instantiate_component(
                g, name, self.config.mass_wasting.kwargs
            )

    def _sync_bedrock(self, g) -> None:
        if "soil__depth" in g.at_node and "bedrock__elevation" in g.at_node:
            g.at_node["bedrock__elevation"][:] = (
                g.at_node["topographic__elevation"] - g.at_node["soil__depth"]
            )

    def _ensure_raster_step_buffers(self, n_nodes: int, n_gospl: int) -> None:
        """Allocate reusable before/Δ/pull buffers once (same shapes each step)."""
        if self._buf_z_before is None or self._buf_z_before.size != n_nodes:
            self._buf_z_before = np.empty(n_nodes, dtype=np.float64)
            self._buf_soil_before = np.empty(n_nodes, dtype=np.float64)
            self._buf_dz = np.empty(n_nodes, dtype=np.float64)
            self._buf_dsoil = np.empty(n_nodes, dtype=np.float64)
        if self._buf_pull is None or self._buf_pull.size != n_gospl:
            self._buf_pull = np.empty(n_gospl, dtype=np.float64)
            self._buf_pull_soil = np.empty(n_gospl, dtype=np.float64)

    def _run_gospl_spl_landlab_hylands(self) -> None:
        """After goSPL SPL: raster DDTD (+ optional landslider); Δz + Δsoil write-back."""
        self._ensure_mirrors()
        self._run_raster_hillslope_mw_dz()

    def _run_raster_hillslope_mw_dz(self) -> None:
        """Push elev → DDTD → D8 → MFD/DF → landslider; write back Δz and Δsoil.

        Remap uses cached barycentric weights (built before the time-step bar).
        DDTD is Landlab NumPy; landslide scar growth stays serial Landlab Python.
        FA / HyLands routing objects are reused (fields refreshed only).
        """
        m = self.model
        z_pre = self._gather_elevation()
        area = self._gather_area()

        if self.rank == 0:
            from .threads import blas_threads_from_env, set_blas_threads

            n_thr = blas_threads_from_env(default=10) or 10
            set_blas_threads(n_thr)

            assert self.mirror is not None
            push_elevation(self.mirror, z_pre)
            g = self.mirror.grid
            self._sync_bedrock(g)

            n_ll = int(g.number_of_nodes)
            self._ensure_raster_step_buffers(n_ll, int(z_pre.size))
            np.copyto(self._buf_z_before, g.at_node["topographic__elevation"])
            np.copyto(self._buf_soil_before, g.at_node["soil__depth"])

            # Hillslope creep first (stable regular dx; no Voronoi short links)
            if self.config.hillslope.is_landlab and self._hillslope_comp is not None:
                dt_sub = float(m.dt) / max(int(self.config.substeps), 1)
                n_sub = max(int(self.config.substeps), 1)
                for _ in range(n_sub):
                    self._call_comp(self._hillslope_comp, dt_sub)
                    self._sync_bedrock(g)

            from .flow_adapter import prepare_hylands_routing

            if self.config.mass_wasting and self.config.mass_wasting.is_landlab:
                if self.config.flow == "landlab":
                    if self._flow_comp is not None:
                        self._call_comp(self._flow_comp, float(m.dt))
                    else:
                        run_landlab_flow(
                            self.mirror,
                            self.config.landlab_flow_component,
                            self.config.landlab_flow_kwargs,
                        )
                else:
                    q, rcv, slope = self._gather_flow_graph()
                    rain = float(np.mean(self._gather_rain()))
                    map_gospl_flow_to_landlab(
                        self.mirror,
                        discharge=q,
                        receiver_steepest=rcv,
                        slope=slope,
                        rainfall=rain if rain > 0 else 1.0,
                    )

                if self._landslide_comp is not None:
                    prepare_hylands_routing(g)
                    self._call_comp(self._landslide_comp, float(m.dt))
                    self._sync_bedrock(g)

            # Re-read fields after Landlab (do not trust pre-step views)
            np.subtract(
                g.at_node["topographic__elevation"],
                self._buf_z_before,
                out=self._buf_dz,
            )
            np.subtract(
                g.at_node["soil__depth"],
                self._buf_soil_before,
                out=self._buf_dsoil,
            )
            dz_gospl = pull_field_to_gospl(
                self.mirror, self._buf_dz, out=self._buf_pull
            )
            z_after = z_pre + dz_gospl
            dsoil_gospl = pull_field_to_gospl(
                self.mirror, self._buf_dsoil, out=self._buf_pull_soil
            )

            outflux, _soil_abs = extract_landlab_sediment(self.mirror.grid)
            if outflux is not None:
                outflux = pull_field_to_gospl(self.mirror, outflux)
            sed_vol = choose_sediment_volume(
                dz=dz_gospl, area=area, landlab_outflux=outflux, dt=float(m.dt)
            )
            # Copy pulled fields for bcast (reuse buffers stay owned by runner)
            payload = (z_after, sed_vol, np.asarray(dsoil_gospl, dtype=np.float64).copy())
        else:
            payload = None

        payload = self.comm.bcast(payload, root=0)
        z_after, sed_vol, dsoil = payload
        self._write_elevation(z_after)
        self._accumulate_cumed(z_pre, z_after)
        if not hasattr(m, "_ll_sed_vol"):
            m._ll_sed_vol = None
        m._ll_sed_vol = sed_vol
        if dsoil is not None and self.config.soil_policy != "off":
            self._add_soil_if_possible(dsoil)

    def _run_landlab_processes(self, stages=None, stage: Optional[str] = None) -> None:
        """Legacy one-raster path (Landlab fluvial ± hillslope/MW)."""
        if stages is None:
            if stage is None:
                raise ValueError("stages or stage required")
            stages = [stage]
        elif isinstance(stages, str):
            stages = [stages]
        stages = list(stages)

        self._ensure_mirrors()
        m = self.model
        z_before = self._gather_elevation()
        area = self._gather_area()

        if self.rank == 0:
            assert self.mirror is not None
            push_elevation(self.mirror, z_before)

            g = self.mirror.grid
            self._sync_bedrock(g)

            from .flow_adapter import prepare_hylands_routing

            if self.config.flow == "landlab":
                if self._flow_comp is not None:
                    self._call_comp(self._flow_comp, float(m.dt))
                else:
                    run_landlab_flow(
                        self.mirror,
                        self.config.landlab_flow_component,
                        self.config.landlab_flow_kwargs,
                    )
            else:
                q, rcv, slope = self._gather_flow_graph()
                rain = float(np.mean(self._gather_rain()))
                map_gospl_flow_to_landlab(
                    self.mirror,
                    discharge=q,
                    receiver_steepest=rcv,
                    slope=slope,
                    rainfall=rain if rain > 0 else 1.0,
                )

            dt_sub = float(m.dt) / max(int(self.config.substeps), 1)
            n_sub = max(int(self.config.substeps), 1)
            creep_stages = [s for s in stages if s != "mass_wasting"]
            for _ in range(n_sub):
                for st in creep_stages:
                    if st == "fluvial" and self._fluvial_comp is not None:
                        self._call_comp(self._fluvial_comp, dt_sub)
                    elif st == "hillslope" and self._hillslope_comp is not None:
                        self._call_comp(self._hillslope_comp, dt_sub)
                    self._sync_bedrock(g)
            if "mass_wasting" in stages and self._landslide_comp is not None:
                prepare_hylands_routing(g)
                self._call_comp(self._landslide_comp, float(m.dt))
                self._sync_bedrock(g)

            z_after = pull_elevation(self.mirror)
            outflux, soil = extract_landlab_sediment(self.mirror.grid)
            if outflux is not None:
                outflux = pull_field_to_gospl(self.mirror, outflux)
            if soil is not None:
                soil = pull_field_to_gospl(self.mirror, soil)
            dz = z_after - z_before
            sed_vol = choose_sediment_volume(
                dz=dz, area=area, landlab_outflux=outflux, dt=float(m.dt)
            )
            payload = (z_after, sed_vol, soil)
        else:
            payload = None

        payload = self.comm.bcast(payload, root=0)
        z_after, sed_vol, soil = payload
        self._write_elevation(z_after)
        self._accumulate_cumed(z_before, z_after)
        if "fluvial" in stages:
            self._set_erosion_flux_for_sedchange(z_before, z_after)
        if not hasattr(m, "_ll_sed_vol"):
            m._ll_sed_vol = None
        m._ll_sed_vol = sed_vol
        if soil is not None and self.config.soil_policy != "off":
            self._write_soil_if_possible(soil)

    def _set_erosion_flux_for_sedchange(self, z_before: np.ndarray, z_after: np.ndarray) -> None:
        """Mirror stock SPL bookkeeping so ``sedChange`` can route Landlab erosion.

        With stratigraphy on, goSPL ``_getSedFlux`` reads ``thCoarse`` from
        ``erodeStrat()``, which itself reads signed elev change in ``tmp``
        (stock: ``tmp = Eb * dt``). Bridge must set ``tmp = dz`` then call
        ``erodeStrat`` — not pre-assign ``thCoarse`` (that gets overwritten).

        ``Eb`` uses stock sign: negative incision, positive deposition.
        """
        m = self.model
        dz = scatter_field(
            (np.asarray(z_after) - np.asarray(z_before)) if self.rank == 0 else None,
            int(m.mpoints),
        )
        locIDs = np.asarray(m.locIDs, dtype=int)
        n_owned = len(m.glIDs) if hasattr(m, "glIDs") else dz.size
        dt = max(float(m.dt), 1e-12)

        # Stock SPL: tmp = Eb*dt = signed thickness this step (neg = incision).
        tmp_local = np.zeros(m.lpoints, dtype=float)
        tmp_local[:n_owned] = dz[locIDs[:n_owned]]
        m.tmpL.setArray(tmp_local)
        m.dm.localToGlobal(m.tmpL, m.tmp)

        # Stock Eb sign: dz/dt (negative where Landlab lowered the surface).
        eb_local = np.zeros(m.lpoints, dtype=float)
        eb_local[:n_owned] = dz[locIDs[:n_owned]] / dt
        m.EbLocal.setArray(eb_local)
        m.dm.localToGlobal(m.EbLocal, m.Eb)

        if getattr(m, "stratNb", 0) and m.stratNb > 0 and hasattr(m, "erodeStrat"):
            # Builds thCoarse (erosion-positive rate) from tmp; do not pre-set
            # thCoarse; do not deposeStrat here (sedChange/seaChange handle depo).
            m.erodeStrat()


    @staticmethod
    def _call_comp(comp, dt: float) -> None:
        mode = getattr(comp, "_gospl_ll_call_mode", None)
        if mode is None:
            if hasattr(comp, "run_one_step"):
                import inspect

                sig = inspect.signature(comp.run_one_step)
                mode = "ros0" if len(sig.parameters) == 0 else "ros1"
            elif hasattr(comp, "update"):
                mode = "update"
            else:
                raise AttributeError(f"{type(comp).__name__} has no run_one_step/update")
            try:
                setattr(comp, "_gospl_ll_call_mode", mode)
            except Exception:
                pass
        if mode == "ros0":
            comp.run_one_step()
        elif mode == "ros1":
            comp.run_one_step(dt)
        else:
            comp.update()

    # --------------------------------------------------------- goSPL field IO
    def _gather_xyz(self):
        m = self.model
        # Prefer global mesh coordinates on rank 0
        if hasattr(m, "mCoords") and m.mCoords is not None:
            coords = np.asarray(m.mCoords)
            if self.rank == 0:
                x, y = coords[:, 0], coords[:, 1]
            else:
                x = y = None
            x = self.comm.bcast(x, root=0)
            y = self.comm.bcast(y, root=0)
        else:
            loc = np.asarray(m.lcoords)
            # serial fallback
            x, y = loc[:, 0], loc[:, 1]
        z = self._gather_elevation()
        return x, y, z

    def _gather_elevation(self) -> np.ndarray:
        m = self.model
        # hGlobal is the parallel Vec; getArray on rank ownership only.
        # Easiest portable path: localToGlobal already done; use local + Allgatherv via locIDs.
        h_local = np.asarray(m.hLocal.getArray(), dtype=float).copy()
        if self.comm.Get_size() == 1:
            # drop ghosts if present: owned nodes
            if hasattr(m, "glIDs"):
                return h_local[np.asarray(m.glIDs)]
            return h_local
        # Broadcast full elevation by placing local owned values into global buffer
        n_global = int(m.mpoints) if hasattr(m, "mpoints") else None
        n_global = self.comm.bcast(n_global, root=0)
        buf = np.zeros(n_global, dtype=float)
        owned = np.asarray(m.glIDs) if hasattr(m, "glIDs") else np.arange(h_local.size)
        gids = np.asarray(m.lgmap_row.apply(owned)) if hasattr(m, "lgmap_row") else owned
        # fallback: locIDs mapping
        if hasattr(m, "locIDs"):
            locIDs = np.asarray(m.locIDs, dtype=int)
            # owned portion of locIDs
            n_owned = len(owned) if hasattr(m, "glIDs") else h_local.size
            buf_local_ids = locIDs[:n_owned]
            buf[buf_local_ids] = h_local[:n_owned]
        else:
            buf[gids] = h_local[owned] if hasattr(m, "glIDs") else h_local
        out = np.zeros_like(buf)
        self.comm.Allreduce(buf, out, op=__import__("mpi4py").MPI.SUM)
        return out

    def _gather_area(self) -> np.ndarray:
        m = self.model
        if hasattr(m, "larea") and self.comm.Get_size() == 1:
            a = np.asarray(m.larea, dtype=float)
            if hasattr(m, "glIDs"):
                return a[np.asarray(m.glIDs)]
            return a.copy()
        # approximate from global area vec if present
        if hasattr(m, "areaGlobal"):
            # gather similarly to elevation — use larea + locIDs
            a_local = np.asarray(m.larea, dtype=float)
            n_global = int(m.mpoints)
            buf = np.zeros(n_global, dtype=float)
            locIDs = np.asarray(m.locIDs, dtype=int)
            n_owned = len(m.glIDs) if hasattr(m, "glIDs") else a_local.size
            buf[locIDs[:n_owned]] = a_local[:n_owned]
            out = np.zeros_like(buf)
            self.comm.Allreduce(buf, out, op=__import__("mpi4py").MPI.SUM)
            return out
        return np.ones(int(m.mpoints), dtype=float)

    def _gather_rain(self) -> np.ndarray:
        m = self.model
        if hasattr(m, "rainVal"):
            r = np.asarray(m.rainVal, dtype=float)
            if self.comm.Get_size() == 1:
                return r.copy()
        return np.array([1.0])

    def _gather_flow_graph(self):
        m = self.model
        # discharge
        q_local = np.asarray(m.FAL.getArray(), dtype=float).copy()
        rcv = np.asarray(getattr(m, "rcvIDi", m.rcvID))
        if rcv.ndim == 2:
            rcv_s = rcv[:, 0].copy()
        else:
            rcv_s = rcv.copy()

        if self.comm.Get_size() == 1:
            if hasattr(m, "glIDs"):
                ids = np.asarray(m.glIDs)
                return q_local[ids], rcv_s[ids], None
            return q_local, rcv_s, None

        # Parallel: pack into global via locIDs (same pattern as elevation)
        n_global = int(m.mpoints)
        locIDs = np.asarray(m.locIDs, dtype=int)
        n_owned = len(m.glIDs)

        q_buf = np.zeros(n_global, dtype=float)
        r_buf = np.full(n_global, -1, dtype=int)
        q_buf[locIDs[:n_owned]] = q_local[:n_owned]
        # map local receiver index → global locID
        r_local = rcv_s[:n_owned].astype(int)
        r_global = np.where(r_local >= 0, locIDs[np.clip(r_local, 0, locIDs.size - 1)], -1)
        r_buf[locIDs[:n_owned]] = r_global

        q_out = np.zeros_like(q_buf)
        r_out = np.zeros_like(r_buf)
        self.comm.Allreduce(q_buf, q_out, op=__import__("mpi4py").MPI.SUM)
        # for receivers use MAX of indices (valid ≥0); -1 fills stay -1 if never set
        self.comm.Allreduce(r_buf, r_out, op=__import__("mpi4py").MPI.MAX)
        return q_out, r_out, None

    def _write_elevation(self, z_global: np.ndarray) -> None:
        m = self.model
        z_global = np.asarray(z_global, dtype=float).ravel()
        z_global = scatter_field(z_global if self.rank == 0 else None, z_global.size)
        h_local = m.hLocal.getArray()
        locIDs = np.asarray(m.locIDs, dtype=int)
        n_owned = len(m.glIDs) if hasattr(m, "glIDs") else min(h_local.size, z_global.size)
        h_local[:n_owned] = z_global[locIDs[:n_owned]]
        m.hLocal.setArray(h_local)
        m.dm.localToGlobal(m.hLocal, m.hGlobal)
        m.dm.globalToLocal(m.hGlobal, m.hLocal)

    def _accumulate_cumed(self, z_before: np.ndarray, z_after: np.ndarray) -> None:
        m = self.model
        dz = np.asarray(z_after - z_before, dtype=float)
        dz = scatter_field(dz if self.rank == 0 else None, dz.size)
        # cumED: positive deposition, negative erosion (goSPL convention in newer; older may differ)
        # Here we store elevation change: z_after - z_before
        c_local = m.cumEDLocal.getArray()
        locIDs = np.asarray(m.locIDs, dtype=int)
        n_owned = len(m.glIDs) if hasattr(m, "glIDs") else min(c_local.size, dz.size)
        c_local[:n_owned] += dz[locIDs[:n_owned]]
        m.cumEDLocal.setArray(c_local)
        m.dm.localToGlobal(m.cumEDLocal, m.cumED)

    def _write_soil_if_possible(self, soil_global: np.ndarray) -> None:
        """Replace goSPL soil with remapped absolute depths (legacy path)."""
        m = self.model
        if not hasattr(m, "Lsoil"):
            if self.rank == 0 and self.config.soil_policy == "require":
                raise RuntimeError("goSPL soil vectors missing but soil_policy=require")
            return
        soil_global = scatter_field(soil_global if self.rank == 0 else None, soil_global.size)
        arr = m.Lsoil.getArray()
        locIDs = np.asarray(m.locIDs, dtype=int)
        n_owned = len(m.glIDs)
        arr[:n_owned] = soil_global[locIDs[:n_owned]]
        m.Lsoil.setArray(arr)
        if hasattr(m, "Gsoil"):
            m.dm.localToGlobal(m.Lsoil, m.Gsoil)

    def _add_soil_if_possible(self, dsoil_global: np.ndarray) -> None:
        """Add remapped Δsoil into goSPL Lsoil (demo Run B raster path)."""
        m = self.model
        if not hasattr(m, "Lsoil"):
            if self.rank == 0 and self.config.soil_policy == "require":
                raise RuntimeError("goSPL soil vectors missing but soil_policy=require")
            return
        dsoil_global = scatter_field(
            dsoil_global if self.rank == 0 else None, dsoil_global.size
        )
        arr = m.Lsoil.getArray()
        locIDs = np.asarray(m.locIDs, dtype=int)
        n_owned = len(m.glIDs)
        arr[:n_owned] += dsoil_global[locIDs[:n_owned]]
        m.Lsoil.setArray(arr)
        if hasattr(m, "Gsoil"):
            m.dm.localToGlobal(m.Lsoil, m.Gsoil)
