import geovista as gv
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as trame_vtk, vuetify
import xarray as xr

from pvxarray.vtk_source import PyVistaXarraySource

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVista Xarray Level of Detail"

# -----------------------------------------------------------------------------
ds = xr.tutorial.load_dataset("air_temperature")
da = ds.air
source = PyVistaXarraySource(da, x="lon", y="lat", time="time", resolution=1.0)

# ds = xr.open_dataset("oisst-avhrr-v02r01.19810901.nc")
# da = ds.err
# source = PyVistaXarraySource(da, x="lon", y="lat", z="zlev", time="time", resolution=0.25)
# source.z_index = 0

# ds = xr.tutorial.load_dataset("eraint_uvz")
# da = ds.u
# source = PyVistaXarraySource(
#     da, x="longitude", y="latitude", z="level", time="month", resolution=1.0
# )
# source.z_index = 0


# store = "https://ncsa.osn.xsede.org/Pangeo/pangeo-forge/pangeo-forge/CMIP6-PMIP-feedstock/CMIP6.PMIP.MIROC.MIROC-ES2L.past1000.r1i1p1f2.Amon.tas.gn.v20200318.zarr"
# ds = xr.open_dataset(store, engine="zarr", chunks={})
# da = ds.tas
# source = PyVistaXarraySource(da, x="lon", y="lat", time="time", resolution=1.0)
# source.z_index = 0
# -----------------------------------------------------------------------------
state.view_edge_visiblity = False

plotter = gv.GeoPlotter(off_screen=True)


def _update():
    source.apply()
    ctrl.view_update()


@state.change("resolution")
def update_resolution(resolution=25, **kwargs):
    source.resolution = resolution / 100.0
    _update()


@state.change("time_index")
def update_time_index(time_index=0, **kwargs):
    source.time_index = time_index
    _update()


@state.change("z_index")
def update_z_index(z_index=0, **kwargs):
    source.z_index = z_index
    source.Update()
    ctrl.view_update()


# Requires https://github.com/pyvista/pyvista/pull/3318 and https://github.com/bjlittle/geovista/pull/127
actor = plotter.add_mesh(
    source,
    cmap="coolwarm",
    show_edges=state.view_edge_visiblity,
    # Requires https://github.com/pyvista/pyvista/pull/3556
    nan_opacity=0,
)
basemap_actor = plotter.add_base_layer(texture=gv.blue_marble())
resolution = "10m"
plotter.add_coastlines(resolution=resolution, color="white")
# plotter.add_axes()
# plotter.add_text(
#     f"NOAA/NCEI OISST AVHRR ({resolution} Coastlines)",
#     position="upper_left",
#     font_size=10,
#     shadow=True,
# )
# plotter.enable_depth_peeling()
plotter.view_isometric()


@state.change("view_edge_visiblity")
def update_edge_visibility(view_edge_visiblity=True, **kwargs):
    actor.GetProperty().SetEdgeVisibility(1 if view_edge_visiblity else 0)
    ctrl.view_update()


@state.change("view_basemap")
def update_view_basemap(view_basemap=True, **kwargs):
    basemap_actor.SetVisibility(1 if view_basemap else 0)
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text("PyVista Xarray Level of Detail")

    with layout.toolbar:
        vuetify.VSpacer()
        max_z = len(da[source.z]) - 1 if source.z else 0
        vuetify.VSlider(
            v_show=max_z,
            v_model=("z_index", 0),
            min=0,
            max=max_z,
            step=1,
            hide_details=True,
            label="Z Index",
            dense=True,
            style="max-width: 300px",
        )
        vuetify.VSlider(
            v_model=("time_index", 0),
            min=0,
            max=len(da[source.time]) - 1,
            step=1,
            hide_details=True,
            label="Time Index",
            dense=True,
            style="max-width: 300px",
        )
        vuetify.VSelect(
            label="Resolution %",
            v_model=("resolution", source.resolution * 100.0),
            items=("array_list", [5, 25, 50, 100]),
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1 ml-2",
            style="max-width: 150px",
        )
        vuetify.VCheckbox(
            v_model=("view_edge_visiblity", True),
            dense=True,
            hide_details=True,
            on_icon="mdi-border-all",
            off_icon="mdi-border-outside",
            classes="ma-2",
        )
        vuetify.VCheckbox(
            v_model=("view_basemap", True),
            dense=True,
            hide_details=True,
            on_icon="mdi-earth",
            off_icon="mdi-earth-off",
            classes="ma-2",
        )

    with (
        layout.content,
        vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ),
    ):
        view = trame_vtk.VtkRemoteView(
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
