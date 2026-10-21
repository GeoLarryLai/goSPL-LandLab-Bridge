# Function documentation

Public functions and classes in `gospl_landlab_bridge`. Each entry is the name, the file, and the first docstring line.

## gospl_landlab_bridge.capability

- `Capability` (gospl_landlab_bridge/capability.py): No docstring.
- `probe_component` (gospl_landlab_bridge/capability.py): Resolve import and return table entry (or experimental if unknown).
- `list_supported` (gospl_landlab_bridge/capability.py): No docstring.

## gospl_landlab_bridge.config

- `SlotSpec` (gospl_landlab_bridge/config.py): One process slot: either goSPL builtin, a Landlab component, or ``off``.
- `BridgeConfig` (gospl_landlab_bridge/config.py): No docstring.

## gospl_landlab_bridge.flow_adapter

- `ensure_flow_fields` (gospl_landlab_bridge/flow_adapter.py): No docstring.
- `map_gospl_flow_to_landlab` (gospl_landlab_bridge/flow_adapter.py): Inject goSPL discharge + single-receiver graph into Landlab node fields.
- `clear_hylands_routing_caches` (gospl_landlab_bridge/flow_adapter.py): Drop cached scratch MFD grids and DepressionFinder instances (tests / remesh).
- `update_hill_flow_from_mfd` (gospl_landlab_bridge/flow_adapter.py): Build HyLands ``hill_flow__*`` from Landlab ``FlowDirectorMFD`` (8-neighbor).
- `update_flood_status` (gospl_landlab_bridge/flow_adapter.py): Set ``flood_status_code`` via Landlab ``DepressionFinderAndRouter``.
- `prepare_hylands_routing` (gospl_landlab_bridge/flow_adapter.py): MFD hill flow + depression flood mask for ``BedrockLandslider``.
- `run_landlab_flow` (gospl_landlab_bridge/flow_adapter.py): Instantiate and run a Landlab flow router on the mirror grid.

## gospl_landlab_bridge.gospl_patches

- `apply_gospl_patches` (gospl_landlab_bridge/gospl_patches.py): Idempotent. Call after ``import gospl`` (e.g. from ``load_model``).

## gospl_landlab_bridge.landlab_dtype

- `resolve_landlab_dtype` (gospl_landlab_bridge/landlab_dtype.py): Map config/name to numpy float32 or float64.
- `patch_component_info_float32` (gospl_landlab_bridge/landlab_dtype.py): Rewrite Landlab component ``_info`` so ``float`` fields accept float32.
- `restore_component_info_dtype` (gospl_landlab_bridge/landlab_dtype.py): Undo float32 ``_info`` rewrite (needed before float64 raster DDTD in same kernel).
- `patch_components_float32` (gospl_landlab_bridge/landlab_dtype.py): No docstring.
- `patch_ddtd_float32` (gospl_landlab_bridge/landlab_dtype.py): Allow DepthDependentTaylorDiffuser (Python) to use float32 node fields.
- `restore_ddtd_float64` (gospl_landlab_bridge/landlab_dtype.py): Restore DDTD ``_info`` for float64 grids (raster path / after float32 toggle).

## gospl_landlab_bridge.loader

- `resolve_landlab_class` (gospl_landlab_bridge/loader.py): Resolve ``Space``, ``landlab.components.Space``, or ``BedrockLandslider``.
- `instantiate_component` (gospl_landlab_bridge/loader.py): No docstring.

## gospl_landlab_bridge.mesh_builder

- `build_regular_flat_mesh` (gospl_landlab_bridge/mesh_builder.py): Write ``{out_prefix}.npz`` (v,c,n,z) and ``{out_prefix}_tect.npz`` (z rock-uplift rates).

## gospl_landlab_bridge.mirror

- `MirrorState` (gospl_landlab_bridge/mirror.py): No docstring.
- `push_weights_ready` (gospl_landlab_bridge/mirror.py): No docstring.
- `pull_weights_ready` (gospl_landlab_bridge/mirror.py): No docstring.
- `build_push_weight_cache` (gospl_landlab_bridge/mirror.py): Build goSPL→raster barycentric weights (own tqdm when ``show_progress``).
- `build_pull_weight_cache` (gospl_landlab_bridge/mirror.py): Build raster→goSPL barycentric weights (own tqdm when ``show_progress``).
- `ensure_remap_weight_caches` (gospl_landlab_bridge/mirror.py): Ensure both push and pull weight tables exist (separate tqdm each).
- `push_elevation` (gospl_landlab_bridge/mirror.py): Push goSPL elevations onto the mirror (direct map or interpolate).
- `pull_elevation` (gospl_landlab_bridge/mirror.py): Sample mirror elevations back to goSPL node order (float64 for goSPL).
- `pull_field_to_gospl` (gospl_landlab_bridge/mirror.py): Map a Landlab-node field onto goSPL node order (float64 for goSPL).
- `resolve_landlab_dtype` (gospl_landlab_bridge/mirror.py): Map config/name to a numpy floating dtype (float32 or float64).
- `build_irregular_mirror` (gospl_landlab_bridge/mirror.py): Build a ``VoronoiDelaunayGrid`` from goSPL ``(x,y)``.
- `build_raster_mirror` (gospl_landlab_bridge/mirror.py): Build a regular ``RasterModelGrid`` and scatter-map from irregular goSPL nodes.

## gospl_landlab_bridge.mpi_io

- `get_comm` (gospl_landlab_bridge/mpi_io.py): No docstring.
- `gather_field` (gospl_landlab_bridge/mpi_io.py): Gather variable-length local arrays to rank 0.
- `scatter_field` (gospl_landlab_bridge/mpi_io.py): Broadcast full array from rank 0, then each rank takes its slice if equal partition.
- `warn_if_large` (gospl_landlab_bridge/mpi_io.py): No docstring.

## gospl_landlab_bridge.postprocess

- `latest_run_dir` (gospl_landlab_bridge/postprocess.py): Prefer ``base`` if it has outputs; else newest ``base_N`` with HDF5 steps.
- `list_gospl_steps` (gospl_landlab_bridge/postprocess.py): No docstring.
- `resolve_gospl_h5` (gospl_landlab_bridge/postprocess.py): Path to ``gospl.{step}.p0.h5``, or a cloud-sync conflicted copy if missing.
- `shared_existing_steps` (gospl_landlab_bridge/postprocess.py): Steps with readable goSPL HDF5 on both runs (repairs cloud-sync conflicts).
- `read_step_time_yr` (gospl_landlab_bridge/postprocess.py): Parse ``Time Value`` from ``xmf/gospl{step}.xmf`` when present.
- `sediment_interfaces_from_thickness` (gospl_landlab_bridge/postprocess.py): Stack positive ``stratH`` under the current surface (foreland-style packages).
- `load_gospl_step` (gospl_landlab_bridge/postprocess.py): Load coordinates + elev/erodep (+ optional stratal) for one output step.
- `pick_progress_steps` (gospl_landlab_bridge/postprocess.py): Pick ``n`` steps spanning the run (early / mid / late).
- `pick_steps_nearest_times` (gospl_landlab_bridge/postprocess.py): Map target times (yr) to nearest available output step indices.
- `pick_steps_every_interval` (gospl_landlab_bridge/postprocess.py): Subsample output steps nearest to 0, interval, 2*interval, … through the run end.
- `mid_transect_indices` (gospl_landlab_bridge/postprocess.py): Node indices along a mid-y east–west band (works for jittered meshes).
- `plot_compare_maps` (gospl_landlab_bridge/postprocess.py): 2×2: elev A|B (no mesh); erodep A|B with mesh overlay.
- `plot_mid_transect` (gospl_landlab_bridge/postprocess.py): Simple elevation line; optional stacked strata elevations along transect.
- `plot_frame_maps` (gospl_landlab_bridge/postprocess.py): One time: 2×2 maps — row1 elev A|B (no mesh), row2 erodep A|B (mesh overlay).
- `plot_progress_maps` (gospl_landlab_bridge/postprocess.py): One 2×2 elev|erodep+mesh figure per progress time.
- `plot_progress_sections` (gospl_landlab_bridge/postprocess.py): One row per time step: mid-transect stratal section for A | B.
- `animate_maps_gif` (gospl_landlab_bridge/postprocess.py): GIF: each frame is 2×2 elev A|B (no mesh) + erodep A|B (mesh overlay).
- `animate_sections_gif` (gospl_landlab_bridge/postprocess.py): GIF of mid-transect stratal sections for every output step.
- `plot_frame_combo` (gospl_landlab_bridge/postprocess.py): One time: 3×2 combo — elev A|B, cumED A|B (± mesh), mid-transect A|B.
- `write_combo_frames` (gospl_landlab_bridge/postprocess.py): Write 3-row combo PNGs for every GIF-interval step (existing HDF5 only).
- `list_combo_frame_pngs` (gospl_landlab_bridge/postprocess.py): Ordered ``compare_combo_frame_XXXXX.png`` under ``out_root/frames``.
- `stitch_combo_mp4` (gospl_landlab_bridge/postprocess.py): Build H.264 MP4 from full-resolution combo PNGs (no downscale).
- `stitch_combo_gif` (gospl_landlab_bridge/postprocess.py): Deprecated alias: writes MP4 (``.mp4``) instead of GIF; no downscale.
- `make_progress_and_gifs` (gospl_landlab_bridge/postprocess.py): Build progress PNGs (+ optional GIFs) under ``out_root``.

## gospl_landlab_bridge.progress

- `silence_stdout` (gospl_landlab_bridge/progress.py): Swallow stdout (goSPL banners) while keeping stderr free for tqdm.
- `landlab_substeps_for_dt` (gospl_landlab_bridge/progress.py): Choose BridgeRunner ``substeps`` so each Landlab inner step ≈ ``target_sub_yr``.
- `load_model` (gospl_landlab_bridge/progress.py): Build ``gospl.model.Model``.
- `estimate_n_steps` (gospl_landlab_bridge/progress.py): Approximate number of outer time-loop iterations left (inclusive of end dump).
- `format_run_done` (gospl_landlab_bridge/progress.py): Short informative one-liner after a finished run.
- `destroy_model` (gospl_landlab_bridge/progress.py): Call ``model.destroy()``; silence goSPL shutdown banners when quiet.

## gospl_landlab_bridge.runner

- `BridgeRunner` (gospl_landlab_bridge/runner.py): Outer time loop mirroring goSPL ``Model.runProcesses`` for goSPL ≥2026

## gospl_landlab_bridge.sediment

- `elevation_delta_to_volume` (gospl_landlab_bridge/sediment.py): Positive volume = deposition; negative = erosion (m³ over the step).
- `extract_landlab_sediment` (gospl_landlab_bridge/sediment.py): Return (outflux, soil_thickness) if present on the Landlab grid.
- `choose_sediment_volume` (gospl_landlab_bridge/sediment.py): B+: prefer Landlab outflux (assumed m³/yr) * dt; else -dz*area for eroded mass

## gospl_landlab_bridge.threads

- `set_blas_threads` (gospl_landlab_bridge/threads.py): Cap BLAS/OpenMP threads for vectorized numpy / scipy work.
- `blas_threads_from_env` (gospl_landlab_bridge/threads.py): Thread count for BLAS / Landlab NumPy work.

## gospl_landlab_bridge.vtk_shim

- `install_vtk_shim` (gospl_landlab_bridge/vtk_shim.py): Put stub modules in ``sys.modules`` for ``vtk`` and ``vtk.util.numpy_support``.
