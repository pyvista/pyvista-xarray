import ipywidgets as widgets
import pyvista as pv
from tqdm import tqdm

from pvxarray.vtk_source import PyVistaXarraySource


def time_controls(
    engine: PyVistaXarraySource, plotter: pv.BasePlotter, continuous_update=False, step=1
):
    def update_time_index(time_index):
        engine.time_index = time_index
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
    return widgets.HBox([play, slider])


def show_ui(engine: PyVistaXarraySource, plotter: pv.BasePlotter, continuous_update=False, step=1):
    iframe = plotter.show(return_viewer=True, jupyter_kwargs={"height": "600px", "width": "99%"})
    controls = time_controls(engine, plotter, continuous_update=continuous_update, step=step)
    return widgets.VBox([iframe, controls])


def save_movie(engine: PyVistaXarraySource, plotter: pv.BasePlotter, filename: str, **kwargs):
    plotter.open_movie(filename, **kwargs)
    for tstep in tqdm(range(engine.max_time_index + 1)):
        engine.time_index = tstep
        plotter.write_frame()
    plotter.mwriter.close()  # close out writer (internal API)
    return filename
