# Load the models
from utils import auto_find_model_files, get_geom_func_from_excel
from plotting_utils import plot_2d

unit_override = None
vel_models_dir = "2dS\\San Pasqual W1\\"
vel_models, unit_str = auto_find_model_files(vel_models_dir, unit_override=unit_override)

geometry_file = "2dS\\San Pasqual W1\\24-channel_VsSurf-ReMi_Geometry_W1.xlsx"
# geometry_file = None
if geometry_file is not None:
    geom_interp_func, peak_elevation = get_geom_func_from_excel(geometry_file)
    y_label = "Elevation, " + unit_str
else:
    geom_interp_func = lambda x: 0.0
    peak_elevation = None
    y_label = "Depth, " + unit_str

_ = plot_2d(
    vel_models=vel_models,
    elevation_func=geom_interp_func,
    peak_elevation=peak_elevation,
    y_max=200,
    res=0.4,
    smoothing_sigma=20,
    title='VsSurf ReMi 2dS™ - San Pasqual W1',
    x_label="Array Dist., " + unit_str,
    y_label=y_label,
    cbar_label="Shear-Wave Velocity, " + unit_str + "/sec",
    cbar_vmin=500,
    cbar_vmax=2200,
    contour_color="k",
    # contours=10,
    # contours=[610, 660, 800, 880, 960, 1040],
    show_plot=True,
)
