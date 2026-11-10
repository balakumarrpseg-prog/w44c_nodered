"""
Generate three temperature heatmap images from Tdata.csv and SensorInfo.csv:
  1. heatmap_T0.png      - CFD-only baseline temperature
  2. heatmap_sensors.png - sensors-only inverse-distance interpolation
  3. heatmap_blended.png - sensors blended onto the CFD baseline

Usage:
    python generate_heatmaps.py [--tdata Tdata.csv] [--sensors SensorInfo.csv] [--outdir .]
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Method-2 envelope length scale: T -> T0 far from all sensors
ENVELOPE_L = 10 #10
# Inverse-distance power; higher = other sensors fade faster near a sensor
INV_DIST_POWER = 1 #1


def load_data(tdata_path, sensors_path):
    tdata_df = pd.read_csv(tdata_path, skipinitialspace=True)
    sensor_df = pd.read_csv(sensors_path, skipinitialspace=True)
    tdata_df.columns = tdata_df.columns.str.strip()
    sensor_df.columns = sensor_df.columns.str.strip()
    return tdata_df, sensor_df


def compute_distances(tdata_df, sensor_df):
    sensor_xy = sensor_df[['x-coordinate', 'y-coordinate']].to_numpy()
    node_xy = tdata_df[['x-coordinate', 'y-coordinate']].to_numpy()
    return np.linalg.norm(node_xy[:, None, :] - sensor_xy[None, :, :], axis=2)


def normalised_inverse_distance_weights(dists, power=INV_DIST_POWER):
    # Each row sums to 1; at a sensor only that sensor gets weight 1, others 0
    d_min = dists.min(axis=1, keepdims=True)
    at_sensor = d_min[:, 0] == 0
    ratio = np.divide(d_min, dists, out=np.zeros_like(dists), where=dists > 0)
    w = ratio ** power
    w[at_sensor] = (dists[at_sensor] == 0)
    w /= w.sum(axis=1, keepdims=True)
    return w, d_min


def compute_sensor_delT(tdata_df, sensor_df, dists):
    nearest_node_idx = dists.argmin(axis=0)
    sensor_T0 = tdata_df['T0'].to_numpy()[nearest_node_idx]
    return sensor_df['T'].to_numpy() - sensor_T0


def compute_blended_T(tdata_df, sensor_df, dists):
    """Method 2: normalised inverse-distance weights with an exponential envelope
    so the rise vanishes (T -> T0) far from all sensors."""
    delT_sensors = compute_sensor_delT(tdata_df, sensor_df, dists)
    w, d_min = normalised_inverse_distance_weights(dists)
    envelope = np.exp(-d_min[:, 0] / ENVELOPE_L)
    increment = envelope * (w * delT_sensors).sum(axis=1)
    return tdata_df['T0'].to_numpy() + increment


def compute_sensor_only_T(sensor_df, dists):
    """Sensor-only interpolation: normalised inverse-distance weights, T0 ignored."""
    w, _ = normalised_inverse_distance_weights(dists)
    return (w * sensor_df['T'].to_numpy()).sum(axis=1)


def save_figure(fig, out_path, retries=6):
    """Save to a temp file, then move it over out_path. Retries if the image is
    briefly busy (e.g. being read by the dashboard), so a script run does not fail."""
    tmp_path = out_path + '.tmp'
    for attempt in range(retries):
        try:
            fig.savefig(tmp_path, format='png', dpi=150, bbox_inches='tight', pad_inches=0.05)
            os.replace(tmp_path, out_path)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def plot_heatmap(tdata_df, sensor_df, values, title, mn, mx, out_path, stamp='', cbar_label='', cmap='hot'):
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(tdata_df['x-coordinate'], tdata_df['y-coordinate'], c=values, cmap=cmap, s=28)
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', location='bottom', pad=0.02,
                        fraction=0.04, shrink=0.4, aspect=40)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=12)
    sc.set_clim(mn, mx)

    ax.scatter(sensor_df['x-coordinate'], sensor_df['y-coordinate'], c='cyan', edgecolors='black',
               marker='^', s=80, label='Sensors')
    for _, row in sensor_df.iterrows():
        ax.annotate(str(int(row['Sensor'])), (row['x-coordinate'], row['y-coordinate']), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=8, fontweight='bold', color='cyan',
                    path_effects=[pe.withStroke(linewidth=1, foreground='black')])

    ax.set_xticks([])  # no ticks / labels on either axis
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left', pad=10)
    if stamp:  # subtle "updated hh:mm am/pm" in grey, bottom-right under the plot
        ax.annotate(f'updated {stamp}', xy=(1, 0), xycoords='axes fraction', xytext=(0, -6),
                    textcoords='offset points', ha='right', va='top', fontsize=10, color='grey')
    ax.set_aspect('equal')
    ax.margins(x=0.03, y=0.08)  # room so edge sensors and labels are not clipped
    fig.tight_layout(pad=0.2)
    save_figure(fig, out_path)  # cropped to content
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # always run with the scripts folder as working directory
    parser.add_argument('--tdata', default=os.path.join(script_dir, 'Tdata.csv'), help='Path to Tdata.csv')
    parser.add_argument('--sensors', default=os.path.join(script_dir, 'SensorInfo.csv'), help='Path to SensorInfo.csv')
    default_out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'static'))
    parser.add_argument('--outdir', default=default_out, help='Directory to save output images')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tdata_df, sensor_df = load_data(args.tdata, args.sensors)
    dists = compute_distances(tdata_df, sensor_df)

    T0 = tdata_df['T0'].to_numpy()
    T_sensors_only = compute_sensor_only_T(sensor_df, dists)
    T_blended = compute_blended_T(tdata_df, sensor_df, dists)

    stamp = datetime.now().strftime('%I:%M %p').lower()  # e.g. 02:15 pm, shown in every title

    plot_heatmap(tdata_df, sensor_df, T0, 'Temperature (CFD only)', 17, 22,
                 os.path.join(args.outdir, 'heatmap_T0.png'), stamp)
    plot_heatmap(tdata_df, sensor_df, T_sensors_only, 'Temperature (sensors only)', 22, 24,
                 os.path.join(args.outdir, 'heatmap_sensors.png'), stamp)
    plot_heatmap(tdata_df, sensor_df, T_blended, 'Temperature (sensors+CFD)', 19, 25,
                 os.path.join(args.outdir, 'heatmap_blended.png'), stamp)


if __name__ == '__main__':
    main()
