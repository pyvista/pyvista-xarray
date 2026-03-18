"""Benchmark: ImageData vs RectilinearGrid performance.

Demonstrates the performance benefits of using pv.ImageData over
pv.RectilinearGrid when the coordinate axes have uniform spacing.
The primary benefit is volume rendering performance.

Usage
-----
    uv run python benchmarks/benchmark_image_data.py

The cells3d xarray tutorial dataset is used as a realistic 3D volume
with uniform spacing on all axes.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pyvista as pv
import xarray as xr

import pvxarray  # noqa: F401 - registers the accessor

pv.OFF_SCREEN = True


def format_times(times):
    """Format timing array as mean +/- std."""
    return f"{times.mean() * 1000:.1f} +/- {times.std() * 1000:.1f} ms"


def benchmark_volume_render(mesh, scalar_name, clim, n_iter=5, warmup=True):
    """Time full volume rendering pipeline: add_volume + screenshot."""
    if warmup:
        pl = pv.Plotter(off_screen=True, window_size=(400, 400))
        pl.add_volume(mesh, scalars=scalar_name, clim=clim, opacity="sigmoid")
        pl.screenshot()
        pl.close()

    times = []
    for _ in range(n_iter):
        pl = pv.Plotter(off_screen=True, window_size=(400, 400))
        start = time.perf_counter()
        pl.add_volume(mesh, scalars=scalar_name, clim=clim, opacity="sigmoid")
        pl.screenshot()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        pl.close()

    return np.array(times)


def benchmark_mesh_creation(da, x, y, z, n_iter=10):
    """Time mesh creation via the pvxarray accessor."""
    _ = da.pyvista.mesh(x=x, y=y, z=z)

    times = []
    for _ in range(n_iter):
        if hasattr(da.pyvista, "_mesh"):
            del da.pyvista._mesh
        start = time.perf_counter()
        mesh = da.pyvista.mesh(x=x, y=y, z=z)
        times.append(time.perf_counter() - start)

    return mesh, np.array(times)


def make_rectilinear(da, x, y, z):
    """Create a RectilinearGrid from the same data (bypassing ImageData optimization)."""
    rg = pv.RectilinearGrid()
    rg.x = da[x].values.astype(float)
    rg.y = da[y].values.astype(float)
    rg.z = da[z].values.astype(float)
    rg[da.name or "data"] = da.values.ravel()
    return rg


def make_image_data(da, x, y, z):
    """Create an ImageData from the same data."""
    xx = da[x].values.astype(float)
    yy = da[y].values.astype(float)
    zz = da[z].values.astype(float)
    im = pv.ImageData(
        origin=(xx[0], yy[0], zz[0]),
        spacing=(np.diff(xx[:2])[0], np.diff(yy[:2])[0], np.diff(zz[:2])[0]),
        dimensions=(len(xx), len(yy), len(zz)),
    )
    im[da.name or "data"] = da.values.ravel()
    return im


def print_table(rows, headers):
    """Print a formatted table."""
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def main():
    print("=" * 70)
    print("PyVista-xarray: ImageData vs RectilinearGrid Benchmark")
    print("=" * 70)
    print()
    print(f"PyVista {pv.__version__}  |  NumPy {np.__version__}  |  xarray {xr.__version__}")
    print()

    # =====================================================================
    # Volume Rendering — the primary motivation for this optimization
    # =====================================================================
    print("=" * 70)
    print("VOLUME RENDERING (primary benefit)")
    print("=" * 70)
    print()

    # --- cells3d ---
    ds = xr.tutorial.load_dataset("cells3d")
    da = ds.images.sel(c="nuclei")
    scalar_name = "images"
    clim = (0, 30000)

    print(f"Dataset: cells3d nuclei channel")
    print(f"  Shape: {da.shape}  ({da.nbytes / 1024 / 1024:.1f} MB)")
    dx = np.diff(da.x.values[:2])[0]
    print(f"  Uniform spacing: {dx:.4f} on all axes")
    print()

    # Accessor auto-detection
    accessor_mesh, _ = benchmark_mesh_creation(da, "x", "y", "z", n_iter=3)
    print(f"  Accessor auto-detects: {type(accessor_mesh).__name__}")
    print()

    # Build both mesh types
    im_mesh = make_image_data(da, "x", "y", "z")
    rg_mesh = make_rectilinear(da, "x", "y", "z")

    n_vol = 5
    im_vol = benchmark_volume_render(im_mesh, scalar_name, clim, n_iter=n_vol)
    rg_vol = benchmark_volume_render(rg_mesh, scalar_name, clim, n_iter=n_vol)

    print("  Volume render (add_volume + render to image):")
    rows = [
        ("ImageData", format_times(im_vol), ""),
        ("RectilinearGrid", format_times(rg_vol), f"{rg_vol.mean() / im_vol.mean():.2f}x slower"),
    ]
    print_table(rows, ("Mesh Type", "Time", ""))
    print()

    # --- Synthetic grids at different sizes ---
    print("-" * 70)
    print("Volume rendering at increasing grid sizes")
    print("-" * 70)
    print()

    vol_rows = []
    for n in [60, 100, 150, 200]:
        synth_data = np.random.randn(n, n, n).astype(np.float32)
        coords = np.linspace(0, 1, n)

        im = pv.ImageData(dimensions=(n, n, n), spacing=(1.0 / n, 1.0 / n, 1.0 / n))
        im["density"] = synth_data.ravel()

        rg = pv.RectilinearGrid(coords, coords, coords)
        rg["density"] = synth_data.ravel()

        n_vol_synth = 3
        im_t = benchmark_volume_render(im, "density", (-2, 2), n_iter=n_vol_synth)
        rg_t = benchmark_volume_render(rg, "density", (-2, 2), n_iter=n_vol_synth)

        ratio = rg_t.mean() / im_t.mean()
        pts = f"{n ** 3:,}"
        vol_rows.append((
            f"{n}^3",
            pts,
            f"{im_t.mean() * 1000:.0f} ms",
            f"{rg_t.mean() * 1000:.0f} ms",
            f"{ratio:.2f}x",
        ))

    print_table(vol_rows, ("Grid", "Points", "ImageData", "RectilinearGrid", "Ratio"))
    print()

    # --- Mapper compatibility ---
    print("-" * 70)
    print("Mapper compatibility")
    print("-" * 70)
    print()

    small_im = pv.ImageData(dimensions=(10, 10, 10))
    small_im["d"] = np.random.randn(small_im.n_points).astype(np.float32)
    small_rg = pv.RectilinearGrid(np.arange(10.0), np.arange(10.0), np.arange(10.0))
    small_rg["d"] = small_im["d"].copy()

    mapper_rows = []
    for mapper_name in ["smart", "gpu", "fixed_point"]:
        for label, mesh in [("ImageData", small_im), ("RectilinearGrid", small_rg)]:
            try:
                pl = pv.Plotter(off_screen=True)
                pl.add_volume(mesh, mapper=mapper_name)
                pl.render()
                pl.close()
                mapper_rows.append((mapper_name, label, "OK"))
            except Exception:
                mapper_rows.append((mapper_name, label, "NOT SUPPORTED"))

    print_table(mapper_rows, ("Mapper", "Mesh Type", "Status"))
    print()
    print("  The 'fixed_point' mapper only supports ImageData.")
    print("  RectilinearGrid is limited to 'smart' and 'gpu' mappers.")
    print()

    # =====================================================================
    # Other operations
    # =====================================================================
    print("=" * 70)
    print("OTHER OPERATIONS")
    print("=" * 70)
    print()

    # Mesh creation
    print("Mesh creation (cells3d):")
    _, im_create = benchmark_mesh_creation(da, "x", "y", "z", n_iter=20)
    print(f"  Accessor (auto ImageData): {format_times(im_create)}")
    print()

    # Memory
    print("Memory usage:")
    mem_rows = []
    for label, m in [("cells3d", (im_mesh, rg_mesh))]:
        im_kb = m[0].actual_memory_size
        rg_kb = m[1].actual_memory_size
        mem_rows.append((label, f"{im_kb} kB", f"{rg_kb} kB", f"{rg_kb - im_kb} kB"))
    print_table(mem_rows, ("Dataset", "ImageData", "RectilinearGrid", "Overhead"))
    print()
    print("  Memory difference is small because data arrays dominate.")
    print("  The structural savings (no coordinate arrays) matter more")
    print("  for VTK's internal pipeline optimization.")
    print()

    # Threshold filter
    print("Threshold filter (cells3d):")
    im_thresh = []
    rg_thresh = []
    for _ in range(5):
        vmin, vmax = im_mesh.get_data_range(scalar_name)
        mid = (vmin + vmax) / 2
        start = time.perf_counter()
        im_mesh.threshold(mid, scalars=scalar_name)
        im_thresh.append(time.perf_counter() - start)
        start = time.perf_counter()
        rg_mesh.threshold(mid, scalars=scalar_name)
        rg_thresh.append(time.perf_counter() - start)

    im_thresh = np.array(im_thresh)
    rg_thresh = np.array(rg_thresh)
    print(f"  ImageData:       {format_times(im_thresh)}")
    print(f"  RectilinearGrid: {format_times(rg_thresh)}")
    print()

    # =====================================================================
    # Summary
    # =====================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("ImageData is automatically used when coordinate axes have uniform")
    print("spacing. Key benefits:")
    print()
    vol_speedup = rg_vol.mean() / im_vol.mean()
    print(f"  1. VOLUME RENDERING: {vol_speedup:.1f}x faster on cells3d")
    print("     VTK's volume mapper handles ImageData more efficiently.")
    print("     The 'fixed_point' mapper is exclusive to ImageData.")
    print()
    print("  2. SEMANTIC CORRECTNESS:")
    print("     ImageData is the natural VTK type for uniform grids.")
    print("     Many VTK algorithms have optimized ImageData code paths.")
    print()
    print("  3. The optimization is AUTOMATIC and TRANSPARENT:")
    print("     Users call .pyvista.mesh() as before. Uniform spacing is")
    print("     detected via np.allclose. Non-uniform grids still use")
    print("     RectilinearGrid.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
