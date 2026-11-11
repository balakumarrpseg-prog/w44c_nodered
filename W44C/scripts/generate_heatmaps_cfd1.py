"""
Generate 9 heatmap images from CFD1_data.csv and SensorInfo.csv
(3 quantities x 3 methods):

    T_cfd.png    T_meas.png    T.png      Temperature (C)
    H_cfd.png    H_meas.png    H.png      Humidity (%)
    CO2_cfd.png  CO2_meas.png  CO2.png    CO2 (ppm)

  X_cfd.png   - CFD-only baseline
  X_meas.png  - sensors-only (measured) inverse-distance interpolation
  X.png       - sensors blended onto the CFD baseline (same method as generate_heatmaps.py)

Units: CFD temperature is Kelvin (converted to C) and CFD humidity is a fraction
(converted to %), so they match the sensor values in SensorInfo.csv (T, H, Co2).

Usage:
    python generate_heatmaps_cfd1.py [--cfd CFD1_data.csv] [--sensors SensorInfo.csv] [--outdir DIR]
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from generate_heatmaps import (ENVELOPE_L, compute_distances,
                               normalised_inverse_distance_weights, plot_heatmap)

# key: (name in title, CFD column, conversion CFD -> sensor units, SensorInfo column, unit, (min, max))
# (min, max) is the colour scale, shared by the 3 images of each quantity so they can be compared
QUANTITIES = {
    'T':   ('Temperature', 'temperature',         lambda v: v - 273.15, 'T',   '°C', (17, 25)),
    'H':   ('Humidity',    'relative-humidity-%', lambda v: v * 100.0,  'H',   '%',   (55, 80)),
    'CO2': ('CO2',         'co2-ppm',             lambda v: v,          'Co2', 'ppm', (380, 700)),
}

# decimals for the measured value shown at each sensor position (instead of the sensor id)
LABEL_DECIMALS = {'T': 1, 'H': 1, 'CO2': 0}

# title suffix: "Temperature_cfd", "Temperature_sensor"; '' = blended plot, whose title is "Temperature (unit)"
METHODS = {'cfd': 'cfd', 'sensors': 'sensor', 'blended': ''}

# colour map: 'hot_r' = reversed 'hot', so larger values are darker ('hot' = larger values lighter)
CMAP = 'hot_r'

# file name = <key><suffix>.png  ->  T_cfd.png, T_meas.png, T.png (same for H and CO2)
FILE_SUFFIX = {'cfd': '_cfd', 'sensors': '_meas', 'blended': ''}


def load_data(cfd_path, sensors_path):
    cfd_df = pd.read_csv(cfd_path, skipinitialspace=True)
    sensor_df = pd.read_csv(sensors_path, skipinitialspace=True)
    cfd_df.columns = cfd_df.columns.str.strip()
    sensor_df.columns = sensor_df.columns.str.strip()
    return cfd_df, sensor_df


def interpolate(cfd_vals, sensor_vals, dists):
    """Return (sensors_only, blended) for one quantity.
    dists: node-to-sensor distances for the sensors that have a value."""
    w, d_min = normalised_inverse_distance_weights(dists)

    sensors_only = (w * sensor_vals).sum(axis=1)

    nearest_node = dists.argmin(axis=0)                 # CFD node closest to each sensor
    delta = sensor_vals - cfd_vals[nearest_node]        # sensor minus CFD at the sensor
    envelope = np.exp(-d_min[:, 0] / ENVELOPE_L)        # rise fades far from all sensors
    blended = cfd_vals + envelope * (w * delta).sum(axis=1)
    return sensors_only, blended


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # always run with the scripts folder as working directory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfd', default=os.path.join(script_dir, 'CFD1_data.csv'), help='Path to CFD1_data.csv')
    parser.add_argument('--sensors', default=os.path.join(script_dir, 'SensorInfo.csv'), help='Path to SensorInfo.csv')
    default_out = os.path.normpath(os.path.join(script_dir, '..', '..', 'static'))
    parser.add_argument('--outdir', default=default_out, help='Directory to save output images')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfd_df, sensor_df = load_data(args.cfd, args.sensors)
    all_dists = compute_distances(cfd_df, sensor_df)

    stamp = datetime.now().strftime('%I:%M %p').lower()  # e.g. 02:15 pm, shown in every title

    for key, (name, cfd_col, convert, sensor_col, unit, (mn, mx)) in QUANTITIES.items():
        # only use sensors that actually have a value for this quantity
        sensor_vals_all = pd.to_numeric(sensor_df[sensor_col], errors='coerce').to_numpy()
        valid = ~np.isnan(sensor_vals_all)
        if not valid.any():
            print(f'Skipping {name}: no sensor values in {sensor_col}')
            continue
        sensors = sensor_df[valid]
        sensor_vals = sensor_vals_all[valid]
        dists = all_dists[:, valid]

        cfd_vals = convert(cfd_df[cfd_col].to_numpy())
        sensors_only, blended = interpolate(cfd_vals, sensor_vals, dists)

        results = {'cfd': cfd_vals, 'sensors': sensors_only, 'blended': blended}
        # value shown at each sensor: the CFD value there on the _cfd image, the measured value otherwise
        cfd_at_sensors = cfd_vals[dists.argmin(axis=0)]  # CFD node closest to each sensor
        label_vals = {'cfd': cfd_at_sensors, 'sensors': sensor_vals, 'blended': sensor_vals}
        for method, values in results.items():
            label = METHODS[method]
            title = f'{name}_{label}' if label else f'{name} ({unit})'
            plot_heatmap(cfd_df, sensors, values, title, mn, mx,
                         os.path.join(args.outdir, f'{key}{FILE_SUFFIX[method]}.png'),
                         stamp, cmap=CMAP, labels=[f'{v:.{LABEL_DECIMALS[key]}f}' for v in label_vals[method]])


if __name__ == '__main__':
    main()
