# Load the models
from plotting_utils import plot_2dp_from_geoct

line_1_path = "PalaLine1.mdl"
line_2_path = "PalaLine2.mdl"
line_3_path = "PalaLine3.mdl"
line_4_path = "PalaLine4.mdl"
unit_str = "ft"
plot_2dp_from_geoct(
    line_2_path,
    travel_time_path="PalaLine2.tt",
    title='Pala Line 1',
    x_label="Array Dist, " + unit_str,
    y_label="Depth, " + unit_str,
    cbar_label="P-Wave Velocity, " + unit_str + "/s",
    smoothing_sigma=10,
    contours=[1500, 2000, 3000, 4000, 5000, 6000],
    cbar_vmin=1200,
    cbar_vmax=5000,
    annotation_color="k",
    cut_negative_depth=True,
    contour_width=0.8,
    interp_res=0.5,
    y_max=40,
    limit_tt_to_plot=False,
)
