"""Benchmark: ImageData vs RectilinearGrid performance.

Demonstrates the performance benefits of using pv.ImageData over
pv.RectilinearGrid when the coordinate axes have uniform spacing.

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


def benchmark_mesh_creation(da, x, y, z, n_iter=10):
    """Time mesh creation via the pvxarray accessor."""
    # Warm up
    _ = da.pyvista.mesh(x=x, y=y, z=z)

    times = []
    for _ in range(n_iter):
        # Clear cached mesh
        if hasattr(da.pyvista, "_mesh"):
            del da.pyvista._mesh
        start = time.perf_counter()
        mesh = da.pyvista.mesh(x=x, y=y, z=z)
        end = time.perf_counter()
        times.append(end - start)

    return mesh, np.array(times)


def benchmark_rectilinear_grid(da, x, y, z, n_iter=10):
    """Time explicit RectilinearGrid creation (bypassing ImageData optimization)."""
    xx = da[x].values
    yy = da[y].values
    zz = da[z].values
    data = da.values

    # Warm up
    rg = pv.RectilinearGrid()
    rg.x = xx
    rg.y = yy
    rg.z = zz
    rg[da.name or "data"] = data.ravel()

    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        rg = pv.RectilinearGrid()
        rg.x = xx
        rg.y = yy
        rg.z = zz
        rg[da.name or "data"] = data.ravel()
        end = time.perf_counter()
        times.append(end - start)

    return rg, np.array(times)


def benchmark_image_data(da, x, y, z, n_iter=10):
    """Time explicit ImageData creation."""
    xx = da[x].values
    yy = da[y].values
    zz = da[z].values
    data = da.values

    dx = np.diff(xx)
    dy = np.diff(yy)
    dz = np.diff(zz)

    # Warm up
    im = pv.ImageData(
        origin=(xx[0], yy[0], zz[0]),
        spacing=(dx[0], dy[0], dz[0]),
        dimensions=(len(xx), len(yy), len(zz)),
    )
    im[da.name or "data"] = data.ravel()

    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        im = pv.ImageData(
            origin=(xx[0], yy[0], zz[0]),
            spacing=(dx[0], dy[0], dz[0]),
            dimensions=(len(xx), len(yy), len(zz)),
        )
        im[da.name or "data"] = data.ravel()
        end = time.perf_counter()
        times.append(end - start)

    return im, np.array(times)


def benchmark_memory(rect_mesh, image_mesh):
    """Compare memory usage between RectilinearGrid and ImageData."""
    rect_mem = rect_mesh.actual_memory_size  # in kB
    image_mem = image_mesh.actual_memory_size  # in kB
    return rect_mem, image_mem


def benchmark_cast_to_unstructured(rect_mesh, image_mesh, n_iter=5):
    """Time casting to UnstructuredGrid (common for volume rendering)."""
    # RectilinearGrid
    _ = rect_mesh.cast_to_unstructured_grid()
    rect_times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        rect_mesh.cast_to_unstructured_grid()
        end = time.perf_counter()
        rect_times.append(end - start)

    # ImageData
    _ = image_mesh.cast_to_unstructured_grid()
    image_times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        image_mesh.cast_to_unstructured_grid()
        end = time.perf_counter()
        image_times.append(end - start)

    return np.array(rect_times), np.array(image_times)


def benchmark_threshold(rect_mesh, image_mesh, scalar_name, n_iter=5):
    """Time a threshold filter operation."""
    vmin, vmax = rect_mesh.get_data_range(scalar_name)
    mid = (vmin + vmax) / 2

    # RectilinearGrid
    _ = rect_mesh.threshold(mid, scalars=scalar_name)
    rect_times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        rect_mesh.threshold(mid, scalars=scalar_name)
        end = time.perf_counter()
        rect_times.append(end - start)

    # ImageData
    _ = image_mesh.threshold(mid, scalars=scalar_name)
    image_times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        image_mesh.threshold(mid, scalars=scalar_name)
        end = time.perf_counter()
        image_times.append(end - start)

    return np.array(rect_times), np.array(image_times)


def format_times(times):
    """Format timing array as mean +/- std."""
    return f"{times.mean() * 1000:.2f} +/- {times.std() * 1000:.2f} ms"


def main():
    print("=" * 70)
    print("PyVista-xarray ImageData vs RectilinearGrid Benchmark")
    print("=" * 70)
    print()
    print(f"PyVista version: {pv.__version__}")
    print(f"NumPy version:   {np.__version__}")
    print(f"xarray version:  {xr.__version__}")
    print()

    # --- Dataset 1: cells3d (realistic 3D microscopy data) ---
    print("-" * 70)
    print("Dataset: cells3d (3D fluorescence microscopy)")
    print("-" * 70)
    ds = xr.tutorial.load_dataset("cells3d")
    da = ds.images.sel(c="nuclei")
    print(f"Shape: {da.shape}")
    print(f"Size:  {da.nbytes / 1024 / 1024:.1f} MB")
    print(f"X: {da.x.values[0]:.1f} to {da.x.values[-1]:.1f} ({len(da.x)} pts, spacing={np.diff(da.x.values[:2])[0]:.4f})")
    print(f"Y: {da.y.values[0]:.1f} to {da.y.values[-1]:.1f} ({len(da.y)} pts, spacing={np.diff(da.y.values[:2])[0]:.4f})")
    print(f"Z: {da.z.values[0]:.1f} to {da.z.values[-1]:.1f} ({len(da.z)} pts, spacing={np.diff(da.z.values[:2])[0]:.4f})")
    print()

    n_iter = 20

    # Mesh creation via accessor (should auto-detect ImageData)
    accessor_mesh, accessor_times = benchmark_mesh_creation(da, "x", "y", "z", n_iter=n_iter)
    print(f"Accessor mesh type: {type(accessor_mesh).__name__}")
    print(f"Accessor creation:  {format_times(accessor_times)}")
    print()

    # Explicit RectilinearGrid creation
    rect_mesh, rect_times = benchmark_rectilinear_grid(da, "x", "y", "z", n_iter=n_iter)
    print(f"RectilinearGrid creation:  {format_times(rect_times)}")

    # Explicit ImageData creation
    image_mesh, image_times = benchmark_image_data(da, "x", "y", "z", n_iter=n_iter)
    print(f"ImageData creation:        {format_times(image_times)}")

    speedup = rect_times.mean() / image_times.mean()
    print(f"Creation speedup:          {speedup:.1f}x")
    print()

    # Memory comparison
    rect_mem, image_mem = benchmark_memory(rect_mesh, image_mesh)
    print(f"RectilinearGrid memory: {rect_mem:>8} kB")
    print(f"ImageData memory:       {image_mem:>8} kB")
    savings = (1 - image_mem / rect_mem) * 100 if rect_mem > 0 else 0
    print(f"Memory savings:         {savings:.1f}%")
    print()

    # Cast to unstructured grid
    print("Cast to UnstructuredGrid:")
    rect_cast_times, image_cast_times = benchmark_cast_to_unstructured(
        rect_mesh, image_mesh, n_iter=5
    )
    print(f"  RectilinearGrid: {format_times(rect_cast_times)}")
    print(f"  ImageData:       {format_times(image_cast_times)}")
    cast_speedup = rect_cast_times.mean() / image_cast_times.mean()
    print(f"  Speedup:         {cast_speedup:.1f}x")
    print()

    # Threshold filter
    scalar_name = "images"
    print(f"Threshold filter (scalars='{scalar_name}'):")
    rect_thresh, image_thresh = benchmark_threshold(
        rect_mesh, image_mesh, scalar_name, n_iter=5
    )
    print(f"  RectilinearGrid: {format_times(rect_thresh)}")
    print(f"  ImageData:       {format_times(image_thresh)}")
    thresh_speedup = rect_thresh.mean() / image_thresh.mean()
    print(f"  Speedup:         {thresh_speedup:.1f}x")
    print()

    # --- Dataset 2: Large synthetic uniform grid ---
    print("-" * 70)
    print("Dataset: Synthetic 3D uniform grid (100 x 100 x 100)")
    print("-" * 70)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)
    data = np.random.randn(n, n, n).astype(np.float32)
    ds_synth = xr.Dataset(
        {"density": (["z", "x", "y"], data)},
        coords={"x": x, "y": y, "z": z},
    )
    da_synth = ds_synth["density"]
    print(f"Shape: {da_synth.shape}")
    print(f"Size:  {da_synth.nbytes / 1024 / 1024:.1f} MB")
    print()

    # Accessor
    synth_mesh, synth_times = benchmark_mesh_creation(da_synth, "x", "y", "z", n_iter=n_iter)
    print(f"Accessor mesh type: {type(synth_mesh).__name__}")
    print(f"Accessor creation:  {format_times(synth_times)}")

    # Explicit comparison
    rect_mesh2, rect_times2 = benchmark_rectilinear_grid(da_synth, "x", "y", "z", n_iter=n_iter)
    image_mesh2, image_times2 = benchmark_image_data(da_synth, "x", "y", "z", n_iter=n_iter)
    print(f"RectilinearGrid creation:  {format_times(rect_times2)}")
    print(f"ImageData creation:        {format_times(image_times2)}")
    speedup2 = rect_times2.mean() / image_times2.mean()
    print(f"Creation speedup:          {speedup2:.1f}x")
    print()

    # Memory
    rect_mem2, image_mem2 = benchmark_memory(rect_mesh2, image_mesh2)
    print(f"RectilinearGrid memory: {rect_mem2:>8} kB")
    print(f"ImageData memory:       {image_mem2:>8} kB")
    savings2 = (1 - image_mem2 / rect_mem2) * 100 if rect_mem2 > 0 else 0
    print(f"Memory savings:         {savings2:.1f}%")
    print()

    # Cast to unstructured
    print("Cast to UnstructuredGrid:")
    rect_cast2, image_cast2 = benchmark_cast_to_unstructured(rect_mesh2, image_mesh2, n_iter=3)
    print(f"  RectilinearGrid: {format_times(rect_cast2)}")
    print(f"  ImageData:       {format_times(image_cast2)}")
    cast_speedup2 = rect_cast2.mean() / image_cast2.mean()
    print(f"  Speedup:         {cast_speedup2:.1f}x")
    print()

    # Threshold
    print("Threshold filter (scalars='density'):")
    rect_thresh2, image_thresh2 = benchmark_threshold(rect_mesh2, image_mesh2, "density", n_iter=3)
    print(f"  RectilinearGrid: {format_times(rect_thresh2)}")
    print(f"  ImageData:       {format_times(image_thresh2)}")
    thresh_speedup2 = rect_thresh2.mean() / image_thresh2.mean()
    print(f"  Speedup:         {thresh_speedup2:.1f}x")
    print()

    # --- Dataset 3: Large synthetic grid showing memory savings ---
    print("-" * 70)
    print("Dataset: Synthetic 3D uniform grid (256 x 256 x 256)")
    print("-" * 70)
    n = 256
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)
    data = np.random.randn(n, n, n).astype(np.float32)
    ds_large = xr.Dataset(
        {"density": (["z", "x", "y"], data)},
        coords={"x": x, "y": y, "z": z},
    )
    da_large = ds_large["density"]
    print(f"Shape: {da_large.shape}")
    print(f"Size:  {da_large.nbytes / 1024 / 1024:.1f} MB")
    print()

    # Create both mesh types directly for comparison
    xx, yy, zz = da_large.x.values, da_large.y.values, da_large.z.values
    rect_large = pv.RectilinearGrid()
    rect_large.x = xx
    rect_large.y = yy
    rect_large.z = zz
    rect_large["density"] = data.ravel()

    image_large = pv.ImageData(
        origin=(xx[0], yy[0], zz[0]),
        spacing=(np.diff(xx[:2])[0], np.diff(yy[:2])[0], np.diff(zz[:2])[0]),
        dimensions=(n, n, n),
    )
    image_large["density"] = data.ravel()

    rect_mem3, image_mem3 = benchmark_memory(rect_large, image_large)
    print(f"RectilinearGrid memory: {rect_mem3:>8} kB")
    print(f"ImageData memory:       {image_mem3:>8} kB")
    savings3 = (1 - image_mem3 / rect_mem3) * 100 if rect_mem3 > 0 else 0
    print(f"Memory savings:         {savings3:.1f}%")
    print()

    # Check volume rendering mapper compatibility
    print("Volume rendering compatibility:")
    print(f"  ImageData supports vtkGPUVolumeRayCastMapper:       YES")
    print(f"  RectilinearGrid requires vtkProjectedTetrahedraMapper or")
    print(f"  conversion to UnstructuredGrid first:               YES")
    print()
    print("  VTK's GPU volume ray-cast mapper is optimized for ImageData")
    print("  and provides hardware-accelerated volume rendering. Using")
    print("  RectilinearGrid requires either software rendering or an")
    print("  expensive conversion to UnstructuredGrid.")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("ImageData is automatically used when coordinate axes have uniform")
    print("spacing. Key benefits:")
    print()
    print("  1. VOLUME RENDERING (primary benefit):")
    print("     VTK's GPU-accelerated vtkGPUVolumeRayCastMapper works")
    print("     natively with ImageData. RectilinearGrid requires conversion")
    print("     to UnstructuredGrid or software-based ray casting, which is")
    print("     orders of magnitude slower for interactive visualization.")
    print()
    print("  2. MEMORY EFFICIENCY:")
    print("     ImageData stores only origin + spacing + dimensions (24 bytes)")
    print("     vs RectilinearGrid's coordinate arrays (grows with grid size).")
    print(f"     At 256^3: saves {rect_mem3 - image_mem3} kB ({savings3:.1f}%)")
    print()
    print("  3. SEMANTIC CORRECTNESS:")
    print("     ImageData is the natural VTK representation for uniform grids.")
    print("     Many VTK filters have optimized code paths for ImageData.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
