import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify
import xarray as xr

from pvxarray.vtk_source import PyVistaXarraySource

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVista Xarray Level of Detail"

# -----------------------------------------------------------------------------
# ds = xr.tutorial.load_dataset("air_temperature")
# da = ds.air[dict(time=0)]  # Select DataArray for a timestep
# source = PyVistaXarraySource(da, x="lon", y="lat")

ds = xr.open_dataset("oisst-avhrr-v02r01.19810901.nc")
da = ds.err[dict(time=0, zlev=0)]
source = PyVistaXarraySource(da, x="lon", y="lat")

# -----------------------------------------------------------------------------
plotter = pv.Plotter(off_screen=True)
# Requires https://github.com/pyvista/pyvista/pull/3318
plotter.add_mesh(source, name="data_array", show_edges=True)
plotter.view_xy()


@state.change("resolution")
def update_resolution(resolution=25, **kwargs):
    source.resolution = resolution / 100.0
    source.Update()
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text("PyVista Xarray Level of Detail")

    with layout.toolbar:
        vuetify.VSpacer()
        # vuetify.VSlider(
        #     v_model=("resolution", 25),
        #     min=5,
        #     max=100,
        #     step=1,
        #     hide_details=True,
        #     label="Resolution",
        #     dense=True,
        #     style="max-width: 300px",
        # )
        vuetify.VSelect(
            label="Resolution %",
            v_model=("resolution", 5),
            items=("array_list", [5, 25, 50, 100]),
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1 ml-2",
            style="max-width: 150px",
        )

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            view = vtk.VtkRemoteView(
                plotter.ren_win,
                ref="view",
                interactive_ratio=1,
            )
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

    # Uncomment following line to hide footer
    # layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
