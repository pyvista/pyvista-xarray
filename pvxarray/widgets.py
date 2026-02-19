"""Interactive Jupyter widgets for time-series visualization."""

import ipywidgets as widgets
import pyvista as pv
from tqdm import tqdm

from pvxarray.vtk_source import PyVistaXarraySource


def time_controls(
    engine: PyVistaXarraySource,
    plotter: pv.BasePlotter,
    continuous_update=False,
    step=1,
    show_label=True,
):
    """Create play/slider widgets to scrub through time steps.

    Parameters
    ----------
    engine : PyVistaXarraySource
        The VTK source with a time dimension.
    plotter : pv.BasePlotter
        The PyVista plotter to re-render on time change.
    continuous_update : bool, default False
        Whether the slider triggers updates while dragging.
    step : int, default 1
        Step size for the play widget.
    show_label : bool, default True
        Display a text label with the current time value.
    """
    label = widgets.Label(value=engine.time_label or "") if show_label else None

    def update_time_index(time_index):
        engine.time_index = time_index
        if label is not None:
            label.value = engine.time_label or ""
        plotter.render()

    tmax = engine.max_time_index

    def set_time(change):
        value = change["new"]
        if value < 0:
            value = 0
        if value >= tmax:
            value = tmax - 1
        update_time_index(value)

    play = widgets.Play(
        value=engine.time_index,
        min=0,
        max=tmax,
        step=step,
        description="Time Index",
    )
    play.observe(set_time, "value")

    slider = widgets.IntSlider(min=0, max=tmax, continuous_update=continuous_update)
    widgets.jslink((play, "value"), (slider, "value"))

    children = [play, slider]
    if label is not None:
        children.append(label)
    return widgets.HBox(children)


def show_ui(engine: PyVistaXarraySource, plotter: pv.BasePlotter, continuous_update=False, step=1):
    """Display the plotter with time controls in a Jupyter notebook."""
    iframe = plotter.show(return_viewer=True, jupyter_kwargs={"height": "600px", "width": "99%"})
    controls = time_controls(engine, plotter, continuous_update=continuous_update, step=step)
    return widgets.VBox([iframe, controls])


def save_movie(engine: PyVistaXarraySource, plotter: pv.BasePlotter, filename: str, **kwargs):
    """Render each time step to a movie file."""
    plotter.open_movie(filename, **kwargs)
    for tstep in tqdm(range(engine.max_time_index + 1)):
        engine.time_index = tstep
        plotter.write_frame()
    plotter.mwriter.close()  # close out writer (internal API)
    return filename
