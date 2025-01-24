import re
import os
import itertools
from datetime import datetime
from fastapi import HTTPException
import ast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def meters_to_feet(val_meters: float) -> float:
    return val_meters * 3.280839895


def feet_to_meters(val_feet: float) -> float:
    return val_feet * 0.3048


def parse_datetime(input_str: str) -> datetime:
    try:
        return datetime.strptime(input_str, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"Could not parse date {input_str} due to an unrecognized or invalid format. "
                                   f"Format must match %Y-%m-%dT%H:%M:%S ex: 2008-09-15T15:53:00")
    except OverflowError:
        raise HTTPException(status_code=400, detail=f"Could not parse date {input_str} due to overflow error.")


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    From https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


# Sequence is defined as
# before-tuple, after-tuple, section
# If after-tuple == before-tuple, then color transitions will be smooth
spectral_seq = [
    (0.15, 0.0, 0.5),
    (0.8, 0.0, 0.8), 1 / 255, (0.8, 0.0, 0.8),
    (0.0, 0.0, 1.0), 64 / 255, (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0), 127 / 255, (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0), 128 / 255, (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0), 192 / 255, (1.0, 1.0, 0.0),
    (0.75, 0.25, 0.0), 254 / 255, (0.75, 0.25, 0.0),
    (0.75, 0.0, 0.0), 255 / 255, (0.75, 0.0, 0.0)
]
spectral_cmap = make_colormap(spectral_seq)

vel_section_seq = [
    (123 / 255, 122 / 255, 230 / 255),
    (102 / 255, 45 / 255, 248 / 255), 35 / 255, (102 / 255, 45 / 255, 248 / 255),
    (99 / 255, 151 / 255, 255 / 255), 50 / 255, (99 / 255, 151 / 255, 255 / 255),
    (7 / 255, 90 / 255, 255 / 255), 65 / 255, (7 / 255, 90 / 255, 255 / 255),
    (0 / 255, 247 / 255, 255 / 255), 100 / 255, (0 / 255, 247 / 255, 255 / 255),
    (10 / 255, 245 / 255, 88 / 255), 130 / 255, (10 / 255, 245 / 255, 88 / 255),
    (76 / 255, 204 / 255, 90 / 255), 150 / 255, (76 / 255, 204 / 255, 90 / 255),
    (154 / 255, 208 / 255, 48 / 255), 165 / 255, (154 / 255, 208 / 255, 48 / 255),
    (168 / 255, 250 / 255, 5 / 255), 180 / 255, (168 / 255, 250 / 255, 5 / 255),
    (254 / 255, 247 / 255, 0 / 255), 200 / 255, (254 / 255, 247 / 255, 0 / 255),
    (255 / 255, 156 / 255, 0 / 255), 215 / 255, (255 / 255, 156 / 255, 0 / 255),
    (255 / 255, 105 / 255, 0 / 255), 235 / 255, (255 / 255, 105 / 255, 0 / 255),
    (255 / 255, 0 / 255, 0 / 255), 254 / 255, (255 / 255, 0 / 255, 0 / 255),
    (165 / 255, 0 / 255, 38 / 255), 255 / 255, (165 / 255, 0 / 255, 38 / 255),
]
vel_section_seq_cmap = make_colormap(vel_section_seq)


def load_vel_model(filename, to_meters_factor=1.0):
    vel = np.loadtxt(filename)
    layer_starts = vel[::2, 0] * to_meters_factor
    layer_stops = vel[1::2, 0] * to_meters_factor
    layer_depth = layer_stops - layer_starts
    layer_density = vel[::2, 1]
    layer_velocity = vel[::2, 3] * to_meters_factor
    temp_data = {
        "Start": layer_starts,
        "Stop": layer_stops,
        "Thickness": layer_depth,
        "Density": layer_density,
        "Velocity": layer_velocity,
    }
    depth_indexer = pd.IntervalIndex.from_arrays(layer_starts, layer_stops, closed="left")
    return pd.DataFrame(temp_data, index=depth_indexer)


def get_single_vel_model_array(vel_model, max_model_layers, flatten=False):
    vel_model_df = vel_model.df
    num_model_layers = len(vel_model_df)
    depth_array = vel_model_df['Stop'].to_numpy()
    vel_array = vel_model_df['Velocity'].to_numpy()
    if num_model_layers < max_model_layers:
        depth_array = np.pad(
            array=depth_array,
            pad_width=(0, max_model_layers - num_model_layers),
            mode='edge',
        )
        vel_array = np.pad(
            array=vel_array,
            pad_width=(0, max_model_layers - num_model_layers),
            mode='edge',
        )
    stacked_array = np.stack([depth_array, vel_array], axis=-1)
    if flatten:
        return stacked_array.flatten()
    return stacked_array


def get_vel_models_array_for_cuda(vel_models):
    num_models = len(vel_models)
    max_model_layers = np.max([vel_model.get_num_layers() for vel_model in vel_models])
    vel_model_array = np.fromiter(
        (get_single_vel_model_array(vel_model, max_model_layers, flatten=False) for vel_model in vel_models),
        dtype=np.dtype((float, (max_model_layers, 2))),
        count=num_models,
    )
    return vel_model_array, max_model_layers


class VelocityModel:
    # Note: All units are in meters!!!!!
    def __init__(self, df, position=0.0, elevation=0.0):
        self.df = df
        self.position = position
        self.max_depth = np.max(self.df['Stop'])
        self.vs30 = self.calc_vs30()
        self.vs100 = self.vs30 * 3.28084
        self.elevation = elevation

    @classmethod
    def from_human_mod_df(cls, human_df, position=0.0):
        thickness_vals = list(human_df['Thickness, km'] * 1000)
        density_vals = list(human_df['Density, g/cc'])
        velocity_vals = list(human_df['Vs, km/s'] * 1000)
        layer_starts = []
        layer_stops = []
        current_depth = 0.0
        for thick in thickness_vals:
            layer_starts.append(current_depth)
            layer_stops.append(current_depth + thick)
            current_depth += thick
        temp_data = {
            "Start": layer_starts,
            "Stop": layer_stops,
            "Thickness": thickness_vals,
            "Density": density_vals,
            "Velocity": velocity_vals,
        }
        depth_indexer = pd.IntervalIndex.from_arrays(layer_starts, layer_stops, closed="left")
        return cls(pd.DataFrame(temp_data, index=depth_indexer), position)

    @classmethod
    def from_file(cls, filename, position=0.0, to_meters_factor=1.0):
        df = load_vel_model(filename, to_meters_factor=to_meters_factor)
        return cls(df, position)

    @classmethod
    def from_evodcinv(cls, ai_model, position=0.0):
        thickness_vals = ai_model[:, 0] * 1000
        density_vals = ai_model[:, 3]
        velocity_vals = ai_model[:, 2] * 1000
        layer_starts = []
        layer_stops = []
        current_depth = 0.0
        for thick in thickness_vals:
            layer_starts.append(current_depth)
            layer_stops.append(current_depth + thick)
            current_depth += thick
        temp_data = {
            "Start": layer_starts,
            "Stop": layer_stops,
            "Thickness": thickness_vals,
            "Density": density_vals,
            "Velocity": velocity_vals,
        }
        depth_indexer = pd.IntervalIndex.from_arrays(layer_starts, layer_stops, closed="left")
        return cls(pd.DataFrame(temp_data, index=depth_indexer), position)

    def save_df(self, filename):
        self.df.to_pickle(filename)

    def set_elevation(self, new_elevation):
        self.elevation = new_elevation

    def set_position(self, new_position):
        self.position = new_position

    def get_velocity(self, depth):
        depth = np.max([0, depth + self.elevation])

        if depth >= self.max_depth:
            return self.df['Velocity'].to_numpy()[-1]
        return self.df.loc[depth]['Velocity']

    def get_num_layers(self):
        return len(self.df)

    def calc_vs30(self):
        numer = 0.0
        denom = 0.0
        row_idx = 0
        current_depth = 0
        while current_depth < 30 and row_idx < len(self.df):
            current_row = self.df.iloc[row_idx]
            current_depth = current_row['Stop']
            if current_depth > 30:
                layer_thickness = 30 - current_row['Start']
            else:
                layer_thickness = current_row['Thickness']
            layer_vel = current_row['Velocity']
            numer += layer_thickness
            denom += layer_thickness / layer_vel
            row_idx += 1
        return numer / denom

    def get_disba_params(self):
        disba_params = {
            'thickness': self.df['Thickness'].to_numpy() * 1.0 / 1000.0,
            'velocity_p': self.df['Velocity'].to_numpy() * 1.0 / 1000.0 * 1.7,
            'velocity_s': self.df['Velocity'].to_numpy() * 1.0 / 1000.0,
            'density': self.df['Density'].to_numpy() * 1.0 / 1000.0,
        }
        return disba_params

    def plot_vel_model(self, plt_kwargs=None):
        x_vals = list(itertools.chain(*zip(self.df['Start'].to_list(), self.df['Stop'].to_list())))
        y_vals = list(itertools.chain(*zip(self.df['Velocity'].to_list(), self.df['Velocity'].to_list())))
        if plt_kwargs is None:
            plt.plot(y_vals, x_vals, )
        else:
            plt.plot(y_vals, x_vals, **plt_kwargs)


model_search_pattern = re.compile(r".*-([0-9]+\.?[0-9]*)([mM]|[fF][tT])-[mM]odel\.txt")


def auto_find_model_files(
        path,
        pattern=model_search_pattern,
        unit_override=None,
):
    # print("Finding models in dir ", path)
    # Create vel models list
    vel_models = []
    unit_str = "m"

    # Get the paths
    filenames = os.listdir(path)
    for filename in filenames:
        # print("\tChecking file: ", filename)
        # Regex filename
        matches = pattern.search(filename)
        if matches is not None and len(matches.groups()) == 2:
            # Extract position and Unit
            position = ast.literal_eval(matches.group(1))
            unit = matches.group(2).lower()
            if unit_override is None:
                if unit == "ft":
                    to_meters_factor = 3.28084
                    unit_str = "ft"
                else:
                    to_meters_factor = 1.0
                    unit_str = "m"
            else:
                if unit_override == "ft":
                    to_meters_factor = 3.28084
                    unit_str = "ft"
                else:
                    to_meters_factor = 1.0
                    unit_str = "m"
            filepath = os.path.join(path, filename)
            vel_model = VelocityModel.from_file(
                filepath,
                position=position,
                to_meters_factor=to_meters_factor,
            )
            vel_models.append(vel_model)
            # print("\t\tMatches! Unit=", unit_str, " Position=", position)
    vel_models = sorted(vel_models, key=lambda model: model.position)
    return vel_models, unit_str


def parse_tt_file(
        tt_path,
        x_lower_limit=None,
        x_upper_limit=None,
):
    shot_data = []
    with open(tt_path, "r") as f:
        header = f.readline().split()
        num_shots = ast.literal_eval(header[0])
        unit_flag = ast.literal_eval(header[1])
        if unit_flag == 1:
            unit_str = "ft"
        else:
            unit_str = "m"
        print(num_shots, unit_str)

        for shot_num in range(num_shots):
            # Read shot data
            shot_header = f.readline().split()
            shot_id = ast.literal_eval(shot_header[0])
            shot_receivers = ast.literal_eval(shot_header[1])
            shot_position = f.readline().split()
            shot_x = ast.literal_eval(shot_position[0])
            shot_z = ast.literal_eval(shot_position[1])

            # Read Receivers, discarding data
            for junk in range(shot_receivers):
                f.readline()

            if x_lower_limit is not None and shot_x < x_lower_limit:
                continue
            if x_upper_limit is not None and shot_x > x_upper_limit:
                continue
            shot_data.append([shot_x, shot_z])
    return np.array(shot_data)

def get_data_from_excel(excel_path):
    xf = pd.ExcelFile(excel_path)
    sheet_names = xf.sheet_names
    sheet_name_regex = re.compile("^Station Coords - N X Y Z[w+a-zA-Z0-9]*")
    candidate_sheet_names = [x for x in sheet_names if sheet_name_regex.search(x)]
    sheet_name = None
    if len(candidate_sheet_names) == 0:
        raise ValueError("No valid sheet names found.")
    elif len(candidate_sheet_names) >= 1:
        sheet_name = candidate_sheet_names[0]
    df = xf.parse(sheet_name=sheet_name, header=2)
    headers = df.columns.values.tolist()
    if headers[0] != "Phone":
        raise ValueError("Headers do not match expected value.")
    x_header = headers[1]
    y_header = headers[2]
    z_header = headers[3]
    x_points = df[x_header]
    y_points = df[y_header]
    z_points = df[z_header]
    return x_points, y_points, z_points


def get_points_on_projection(start_point, end_point, to_project):
    def get_point_on_projection(point_to_project):
        # Based on solution here: https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
        # Find distance between start and end points
        l2 = np.sum((start_point - end_point) ** 2)
        if l2 == 0:
            raise ValueError("Start and end points are the same point!")
        t = np.sum((point_to_project - start_point) * (end_point - start_point)) / l2
        return start_point + t * (end_point - start_point)

    return np.apply_along_axis(get_point_on_projection, 1, to_project)


def calc_array_distance(initial_point, distance_points):
    return np.linalg.norm(initial_point - distance_points, axis=1)


def get_geom_func_from_excel(excel_path, method: str = "naive"):
    method = method.lower()

    # Extract raw values
    x_points, y_points, z_points = get_data_from_excel(excel_path)
    xy_points = np.column_stack([x_points, y_points])
    xyz_points = np.column_stack([x_points, y_points, z_points])

    # Get peak elevation
    peak_elevation = np.max(z_points)
    depths = z_points - peak_elevation

    positions = None
    if method == "naive":
        positions = calc_array_distance(xy_points[0], xy_points)
    elif method == "2d_fit":
        fit_line_x = np.unique(x_points)
        fit_line_y = np.poly1d(np.polyfit(x_points, y_points, 1))(np.unique(x_points))
        fit_line = np.column_stack([fit_line_x, fit_line_y])
        projected_points = get_points_on_projection(fit_line[0], fit_line[-1], xy_points)
        projected_points_x = projected_points[:, 0]
        projected_points_y = projected_points[:, 1]
        positions = calc_array_distance(projected_points[0], projected_points)
    elif method == "3d_fit":
        raise NotImplementedError()
    else:
        raise ValueError("Method must be 'naive' or '2d_fit'")

    # Generate geometry function
    geom_interp_func = lambda x: np.interp(x, positions, depths)
    return geom_interp_func, peak_elevation
