import ast
import io
import math
import time
from multiprocessing import Pool

import matplotlib.patches
import numpy as np
import numba as nb
import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule
from os import PathLike
from typing import Union, Optional, Any
from scipy.interpolate import interpolate
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import ndimage
from utils import vel_section_seq_cmap, parse_tt_file, VelocityModel, get_vel_models_array_for_cuda


def fancy_round(num, base):
    return base * round(num / base)


def fancy_round_down(num, base):
    return base * np.floor(num / base)


def calc_model_idxs_and_weights(model_positions_array, x_val):
    less_than = np.where(model_positions_array <= x_val)[0]
    greater_than = np.where(model_positions_array >= x_val)[0]
    min_less = less_than[-1] if len(less_than) >= 1 else None
    max_great = greater_than[0] if len(greater_than) >= 1 else None
    if min_less is not None and max_great is None:
        max_great = min_less
    elif max_great is not None and min_less is None:
        min_less = max_great
    elif min_less is None and max_great is None:
        raise ValueError("Min and Max both invalid. This shouldn't be possible!")
    model_dist = model_positions_array[max_great] - model_positions_array[min_less]
    if model_dist <= 0:
        weight_less = 0.0
        weight_great = 1.0
    else:
        weight_less = (model_positions_array[max_great] - x_val) / model_dist
        weight_great = (x_val - model_positions_array[min_less]) / model_dist
    return min_less, max_great, weight_less, weight_great


def calc_model_velocity_array(vel_model, y_vals, elevation=0.0):
    return np.fromiter((vel_model.get_velocity(y + elevation) for y in y_vals), float)


def calc_column(local_y_shift_array, model_interp_array, vel_models, y_vals, x_ind):
    local_y_shift = local_y_shift_array[x_ind]
    interp_array = model_interp_array[x_ind]
    model_1_ind = int(interp_array[0])
    model_1_vels = calc_model_velocity_array(vel_models[model_1_ind], y_vals, elevation=local_y_shift)
    model_1_weight = interp_array[2]
    model_1_weighted = model_1_weight * model_1_vels
    model_2_ind = int(interp_array[1])
    model_2_vels = calc_model_velocity_array(vel_models[model_2_ind], y_vals, elevation=local_y_shift)
    model_2_weight = interp_array[3]
    model_2_weighted = model_2_weight * model_2_vels
    vel_array = model_1_weighted + model_2_weighted
    return vel_array


def cuda_2ds_array(
        int_arg_array,
        y_vals,
        y_shift_array,
        model_interp_array,
        vel_models,
):
    num_y_points = int_arg_array[2]
    num_x_points = int_arg_array[3]
    total_points = num_y_points * num_x_points
    kernel_code = open('cuda/2ds_array.cuda', 'r').read()
    cuda_mod = SourceModule(kernel_code)
    cuda_func = cuda_mod.get_function("calc_2ds")

    results_array = np.zeros((num_y_points, num_x_points), dtype=np.float64).flatten()
    cuda_block_tuple = (1024, 1, 1)
    cuda_grid_tuple = (math.ceil(total_points / 1024.0), 1)
    cuda_func(
        cuda.InOut(results_array),
        cuda.In(int_arg_array),
        cuda.In(y_vals),
        cuda.In(y_shift_array),
        cuda.In(model_interp_array),
        cuda.In(vel_models),
        block=cuda_block_tuple,
        grid=cuda_grid_tuple
    )
    return results_array


def plot_2d(
        vel_models: list[VelocityModel],
        geom_positions=None,
        elevation_func=None,
        x_min=None,
        x_max=None,
        x_min_trim=None,
        x_max_trim=None,
        y_min=None, y_max=None,
        res=0.1, smoothing_sigma=20,
        contours=None,
        title=None, y_label=None, x_label=None, cbar_label=None,
        cbar_vmin=None,
        cbar_vmax=None,
        contour_color="k",
        contour_width=0.8,
        label_pad_size=-58,
        cbar_pad_size: float = 0.10,
        colorbar=True,
        invert_colorbar_axis=False,
        show_plot=True,
        save_path: Union[str, bytes, PathLike, None] = None,
        aboveground_color: any = "w",
        aboveground_border_color: Optional[Any] = None,
        shift_elevation=False,
        peak_elevation=None,
        cbar_ticks=None,
        reverse=False,
        elevation_tick_increment=50,
        ticks=None,
        ticklabels=None,
        x_label_position="top",
        y_label_position="left"
):
    # Ensure tick / label and ticklabel are all valid
    if ticks is None:
        ticks = {"right": False, "left": True, "top": True, "bottom": False}
    if ticklabels is None:
        ticklabels = {"labelright": False, "labelleft": True, "labeltop": True, "labelbottom": False}

    # Ensure vel models is sorted by position
    vel_models = sorted(vel_models, key=lambda vel_model: vel_model.position)

    # Calculate x_min and x_max if not specified, based off array spacing
    if x_min is None and len(vel_models) >= 2:
        x_min = max(0, vel_models[0].position - abs(vel_models[0].position - vel_models[1].position))
    if x_max is None and len(vel_models) >= 2:
        x_max = vel_models[-1].position + abs(vel_models[-1].position - vel_models[-2].position)
    if y_min is None:
        y_min = 0
    if y_max is None:
        y_max = max([model.max_depth for model in vel_models])
        print("Warning: No max depth specified - using max depth of the provided velocity models, " + str(y_max)
              + ". This is probably WAY too deep! Specify this value using y_max=##")
    if res is None:
        raise ValueError("Resolution (res) cannot be none!")

    x_vals = np.arange(start=x_min, stop=x_max + res, step=res)
    y_vals = np.arange(start=y_min, stop=y_max + res, step=res)
    zz = np.zeros((len(y_vals), len(x_vals)))

    shift_elevation_amount = 0
    if elevation_func is not None:
        # Find the highest point within x range
        max_elev = elevation_func(x_vals[0])
        for x_pt in x_vals:
            elev = elevation_func(x_pt)
            if elev > max_elev:
                max_elev = elev
        if shift_elevation:
            shift_elevation_amount = max_elev
    if peak_elevation is not None:
        peak_elevation += shift_elevation_amount

    # X = point along array
    # Y = depth
    # Fill in grid
    # Pre-calculate arrays
    y_shift_array = np.fromiter((elevation_func(xi) - shift_elevation_amount for xi in x_vals),
                                dtype=np.float64,
                                count=len(x_vals))
    vel_model_positions_array = np.fromiter((vm.position for vm in vel_models),
                                            dtype=np.float64,
                                            count=len(vel_models))
    model_interp_array = np.fromiter((calc_model_idxs_and_weights(vel_model_positions_array, x) for x in x_vals),
                                     dtype=np.dtype((np.float64, 4)),
                                     count=len(x_vals))

    vel_models_array, max_model_layers = get_vel_models_array_for_cuda(vel_models)
    cuda_int_args = np.array([
        len(vel_models),  # num_models
        max_model_layers,
        len(y_vals),
        len(x_vals),
    ], dtype=np.int32)
    start = time.time()
    results_array = cuda_2ds_array(
        int_arg_array=cuda_int_args.astype(np.int32),
        y_vals=y_vals.astype(np.float64),
        y_shift_array=y_shift_array.astype(np.float64),
        model_interp_array=model_interp_array.flatten().astype(np.float64),
        vel_models=vel_models_array.flatten().astype(np.float64)
    )
    # zz_actual_cuda_style = results_array.reshape((len(y_vals), len(x_vals)))
    zz = results_array.reshape((len(y_vals), len(x_vals)))
    end = time.time()
    print(f"Calculated actual cuda in {end - start}")

    # start = time.time()
    # pool = Pool(processes=10)
    # columns = [pool.apply_async(calc_column, [y_shift_array, model_interp_array, vel_models, y_vals, val])
    #            for val in range(len(x_vals))]
    # for idx, val in enumerate(columns):
    #     zz[:, idx] = val.get()
    # end = time.time()
    # print(f"Calculated default in {end - start}")
    #
    # diff_actual_cuda = zz - zz_actual_cuda_style
    # print("Max diff actual cuda: ", np.max(diff_actual_cuda))
    # print("Num diff actual cuda: ", np.count_nonzero(diff_actual_cuda > 1.0))

    if smoothing_sigma > 0:
        smoothed = gauss_with_nan(arr=zz, sigma=smoothing_sigma)
    else:
        smoothed = zz
    if cbar_vmin is None:
        cbar_vmin = np.nanmin(smoothed)
    if cbar_vmax is None:
        cbar_vmax = round(np.nanmax(smoothed) + 7, 2)
    smoothed = np.nan_to_num(smoothed, nan=-1, posinf=-1, neginf=-1)

    if x_min_trim is not None and x_max_trim is not None:
        x_min_trim = max(x_min, x_min_trim)
        x_max_trim = min(x_max, x_max_trim)
        x_min_ind = np.argmin(np.abs(x_vals - x_min_trim))
        x_max_ind = np.argmin(np.abs(x_vals - x_max_trim))
        smoothed = smoothed[:, x_min_ind:x_max_ind]
        x_vals = x_vals[x_min_ind:x_max_ind]

    # If geometry, trim stuff
    if elevation_func is not None:
        for x_ind, x_val in enumerate(x_vals):
            y_val = np.abs(elevation_func(x_val)) + shift_elevation_amount
            y_query_arr = y_vals[y_vals - y_val < 0]
            if y_query_arr.size > 0:
                nearest_y_val = np.max(y_query_arr)
                y_ind = np.min(np.argwhere(y_vals == nearest_y_val)[0])
                smoothed[0:y_ind, x_ind] = np.nan
    fig, ax = plt.subplots(figsize=(10, 5))
    if reverse:
        smoothed = np.flip(smoothed, axis=1)
    mesh = ax.pcolormesh(x_vals, y_vals, smoothed, cmap=vel_section_seq_cmap, vmin=cbar_vmin, vmax=cbar_vmax)
    axes = mesh.axes
    ax.set_ylim([np.nanmin(y_vals), np.nanmax(y_vals)])
    ax.set_xlim([np.nanmin(x_vals), np.nanmax(x_vals)])
    axes.invert_yaxis()

    # Contours
    if contours is not None:
        contour_plot = ax.contour(x_vals, y_vals, smoothed, contours, colors=contour_color, linewidths=contour_width)
        ax.clabel(contour_plot, contour_plot.levels, inline=True, fontsize=10)

    if colorbar:
        cbar = plt.colorbar(mesh, pad=cbar_pad_size, fraction=0.05, orientation="vertical")
        cbar.ax.set_ylabel(cbar_label, labelpad=label_pad_size)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        if invert_colorbar_axis:
            cbar.ax.invert_yaxis()

    if y_label is not None:
        plt.ylabel(y_label)
        ax.yaxis.set_label_position(y_label_position)
    if x_label is not None:
        plt.xlabel(x_label)
        ax.xaxis.set_label_position(x_label_position)
    ax.tick_params(**dict(ticks, **ticklabels))
    if peak_elevation is not None:
        # Initial tick must be 0, peak_elevation
        # Get next elevation tick in elevation
        next_elev_tick = fancy_round_down(peak_elevation, elevation_tick_increment)
        initial_diff = peak_elevation - next_elev_tick

        # If difference is too small, then skip the first tick by increasing initial diff
        if initial_diff < 0.25 * elevation_tick_increment:
            print(initial_diff, next_elev_tick)
            initial_diff += elevation_tick_increment
            next_elev_tick -= elevation_tick_increment

        # Generate new tick locations and labels
        new_ticks = [0, initial_diff]
        new_labels = [int(np.rint(peak_elevation)), int(np.rint(next_elev_tick))]
        next_tick = initial_diff + elevation_tick_increment
        next_elev = next_elev_tick - elevation_tick_increment
        while next_tick <= y_max:
            new_ticks.append(next_tick)
            new_labels.append(int(np.rint(next_elev)))
            next_tick = next_tick + elevation_tick_increment
            next_elev = next_elev - elevation_tick_increment
        ax.set_yticks(new_ticks, labels=new_labels)

    if elevation_func is not None:
        patch_points_list = [[x_pt, np.abs(elevation_func(x_pt)) + shift_elevation_amount] for x_pt in x_vals]
        patch_points_array = np.array(patch_points_list)
        if reverse:
            patch_points_array[:, 0] = np.flip(x_vals)
        # Add point above y-axis at the beginning
        new_point = patch_points_array[0].copy()
        new_point[1] = -100
        patch_points_array = np.insert(
            patch_points_array,
            obj=0,
            values=[new_point, ],
            axis=0
        )
        # Add point above y-axis at the end
        new_point = patch_points_array[-1].copy()
        new_point[1] = -100
        patch_points_array = np.insert(
            patch_points_array,
            obj=patch_points_array.shape[0],
            values=[new_point, ],
            axis=0
        )
        border = matplotlib.patches.Polygon(
            xy=patch_points_array,
            color=aboveground_color,
            edgecolor=aboveground_border_color
        )

        ax.add_patch(border)

    plt.title(title)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show_plot:
        plt.show()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.clf()
    return zz, img_buf


def gauss_with_nan(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)
    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
        loss, sigma=sigma)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(gauss, sigma=sigma)
    gauss[nan_msk] = np.nan
    loss_factor = loss * arr
    gauss += loss_factor

    return gauss


def plot_2dp_from_geoct(
        geoct_model_path: Union[str, bytes, PathLike],
        travel_time_path: Union[str, bytes, PathLike, None] = None,
        x_min: Optional[float] = 0.0,
        x_max: Optional[float] = None,
        y_min: Optional[float] = 0.0,
        y_max: Optional[float] = None,
        smoothing_sigma: Optional[float] = 0,
        contours: Union[list[float], int, None] = None,
        title: Optional[str] = None,
        y_label: Optional[str] = None,
        x_label: Optional[str] = None,
        cbar_label: Optional[str] = None,
        cbar_vmin: Optional[float] = None,
        cbar_vmax: Optional[float] = None,
        annotation_color: any = "k",
        interp_res: Optional[float] = None,
        label_pad_size: float = -58,
        cbar_pad_size: float = 0.10,
        contour_width: float = 0.8,
        limit_tt_to_plot: bool = True,
        poly_color: any = "w",
        poly_border_color: Optional[Any] = None,
        save_path: Union[str, bytes, PathLike, None] = None,
        show_plot: bool = True,
        reverse_data: bool = False,
        show_max_survey_depth: bool = False,
        valid_survey_depth_override: Optional[float] = None,
        ticks_left: bool = True,
        ticks_right: bool = True,
        ticks_top: bool = True,
        ticks_bottom: bool = True,
        labels_left: bool = True,
        labels_right: bool = False,
        labels_top: bool = True,
        labels_bottom: bool = False,
        plot_size=(10, 5),
        aspect_ratio=2.0,
        colorbar=True,
        invert_colorbar_axis=False,
):
    if interp_res is not None and interp_res <= 0:
        interp_res = None

    with open(geoct_model_path, "r") as f:
        header = f.readline()
        header_list = [ast.literal_eval(x) for x in header.split()]
        data_list = []
        for line in f:
            data = [ast.literal_eval(x) for x in line.split()]
            data_list.extend(data)
    zz = np.array(data_list).reshape(header_list[1], header_list[0])
    x_start = header_list[3]
    x_stop = header_list[3] + header_list[0] * header_list[2]
    x_step = header_list[2]
    x_vals = np.arange(
        start=x_start,
        stop=x_stop,
        step=x_step,
    )
    if x_vals.shape[0] > header_list[0]:
        x_vals = x_vals[:header_list[0]]
    y_start = header_list[4]
    y_stop = header_list[4] + header_list[1] * header_list[2]
    y_step = header_list[2]
    y_vals = np.arange(
        start=y_start,
        stop=y_stop,
        step=y_step,
    )

    if y_vals.shape[0] > header_list[1]:
        y_vals = y_vals[:header_list[1]]

    border = None
    travel_time_data = None
    interp_line = None
    clipping_data = None
    # If tt, plot it
    if travel_time_path is not None:
        travel_time_data = parse_tt_file(
            travel_time_path,
            x_lower_limit=np.nanmin(x_vals) if limit_tt_to_plot else None,
            x_upper_limit=np.nanmax(x_vals) if limit_tt_to_plot else None,
        )
        # Check if we need to extend to reach x=xmin
        if travel_time_data[0][0] > np.nanmin(x_vals):
            new_point = travel_time_data[0].copy()
            new_point[0] = np.nanmin(x_vals)
            travel_time_data = np.insert(
                travel_time_data,
                obj=0,
                values=[new_point, ],
                axis=0
            )

        # Check if we need to extend to reach x=max
        if travel_time_data[-1][0] < np.nanmax(x_vals):
            new_point = travel_time_data[-1].copy()
            new_point[0] = np.nanmax(x_vals)
            travel_time_data = np.insert(
                travel_time_data,
                obj=travel_time_data.shape[0],
                values=[new_point, ],
                axis=0
            )

        clipping_data = travel_time_data.copy()

        if reverse_data:
            x_max = np.nanmax(x_vals) if x_max is None else x_max
            travel_time_data[:, 0] = x_max - travel_time_data[:, 0]

        # Add point above y-axis at the beginning
        new_point = travel_time_data[0].copy()
        new_point[1] = -100
        travel_time_data = np.insert(
            travel_time_data,
            obj=0,
            values=[new_point, ],
            axis=0
        )
        # Add point above y-axis at the end
        new_point = travel_time_data[-1].copy()
        new_point[1] = -100
        travel_time_data = np.insert(
            travel_time_data,
            obj=travel_time_data.shape[0],
            values=[new_point, ],
            axis=0
        )

        border = matplotlib.patches.Polygon(
            xy=travel_time_data,
            color=poly_color,
            edgecolor=poly_border_color
        )

    if interp_res is not None:
        interp_func = interpolate.RectBivariateSpline(y_vals, x_vals, zz)
        x_vals = np.arange(start=0, stop=np.nanmax(x_vals) + interp_res, step=interp_res)
        if y_max is not None:
            y_max = min(y_max, np.nanmax(y_vals))
            y_vals = np.arange(start=0, stop=y_max + interp_res, step=interp_res)
        else:
            y_max = np.nanmax(y_vals)
            y_vals = np.arange(start=0, stop=np.nanmax(y_vals) + interp_res, step=interp_res)
        zz = interp_func(y_vals, x_vals)

    if travel_time_data is not None:
        interp_line = np.interp(x_vals, clipping_data[:, 0], clipping_data[:, 1])

        # We now have a line that that spans the plot - use this to filter the meshgrid
        for i in range(interp_line.shape[0]):
            x_ind = int(i)
            y_val = interp_line[i]
            y_ind = int(np.argmin(np.abs(y_vals - y_val)))
            zz[:y_ind, x_ind:x_ind + 1] = np.nan
    if smoothing_sigma > 0:
        smoothed = gauss_with_nan(arr=zz, sigma=smoothing_sigma)
    else:
        smoothed = zz

    # Trim data based on min/max x and y
    num_left_x = np.count_nonzero(x_vals[np.where(x_vals < x_min)]) if x_min is not None else 0
    num_right_x = np.count_nonzero(x_vals[np.where(x_vals > x_max)]) if x_max is not None else 0
    num_top_y = np.count_nonzero(x_vals[np.where(y_vals < y_min)]) if y_min is not None else 0
    num_bot_y = np.count_nonzero(x_vals[np.where(y_vals > y_max)]) if y_max is not None else 0
    smoothed = smoothed[num_top_y:y_vals.shape[0] - num_bot_y, num_left_x:x_vals.shape[0] - num_right_x]
    x_vals = x_vals[num_left_x:x_vals.shape[0] - num_right_x]
    y_vals = y_vals[num_top_y:y_vals.shape[0] - num_bot_y]

    if reverse_data:
        smoothed = np.flip(smoothed, axis=1)

    if cbar_vmin is None:
        cbar_vmin = np.nanmin(smoothed)
    if cbar_vmax is None:
        cbar_vmax = round(np.nanmax(smoothed) + 7, 2)
    # fig, ax = plt.subplots(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=plot_size)
    vel_section_seq_cmap.set_under("white")
    mesh = ax.pcolormesh(x_vals, y_vals, smoothed, cmap=vel_section_seq_cmap, vmin=cbar_vmin, vmax=cbar_vmax)
    axes = mesh.axes
    ax.set_ylim([np.nanmin(y_vals), np.nanmax(y_vals)])
    ax.set_xlim([np.nanmin(x_vals), np.nanmax(x_vals)])
    axes.invert_yaxis()

    # Contours
    if contours is not None:
        contour_plot = ax.contour(x_vals, y_vals, smoothed, contours, colors=annotation_color, linewidths=contour_width)
        ax.clabel(contour_plot, contour_plot.levels, inline=True, fontsize=10)

    if border is not None:
        ax.add_patch(border)

    if valid_survey_depth_override is not None:
        valid_survey_depth = valid_survey_depth_override
    else:
        valid_survey_depth = (np.nanmax(x_vals) - np.nanmin(x_vals)) / 6.0

    if show_max_survey_depth:
        if y_max is None:
            y_max = np.nanmax(y_vals)
        if y_max > valid_survey_depth:
            # Create box
            min_x_point = np.nanmin(x_vals)
            max_x_point = np.nanmax(x_vals)
            box_points = np.array([
                [min_x_point, y_max],
                [max_x_point, y_max],
                [max_x_point, valid_survey_depth],
                [min_x_point, valid_survey_depth],
            ])
            box = matplotlib.patches.Polygon(
                xy=box_points,
                facecolor='0.8',
                edgecolor='k',
                alpha=0.6,
                linewidth=1,
            )
            rx = min_x_point
            ry = valid_survey_depth
            cx = rx + (max_x_point - min_x_point) / 2.0
            cy = ry + (y_max - valid_survey_depth) / 2.0
            ax.add_patch(box)
            ax.annotate("TEST TEXT", (cx, cy), color='k')

    if labels_right:
        cbar_pad_size += 0.05

    if colorbar:
        cbar = plt.colorbar(mesh, pad=cbar_pad_size, orientation="vertical")
        cbar.ax.set_ylabel(cbar_label, labelpad=label_pad_size)
        if invert_colorbar_axis:
            cbar.ax.invert_yaxis()
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    ax.xaxis.set_label_position("bottom")
    ax.set_facecolor("white")
    plt.title(title, pad=20)
    plt.gca().set_aspect(aspect_ratio)

    ax.tick_params(
        bottom=ticks_bottom,
        top=ticks_top,
        left=ticks_left,
        right=ticks_right,
        labelbottom=labels_bottom,
        labeltop=labels_top,
        labelleft=labels_left,
        labelright=labels_right,
    )
    # ticks_left = True,
    # ticks_right = True,
    # ticks_top = True,
    # ticks_bottom = True,
    # labels_left = True,
    # labels_right = False,
    # labels_top = True,
    # labels_bottom = False,

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show_plot:
        plt.show()

    plt.clf()
