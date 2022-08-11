import itertools
import json
import os
import time
from glob import glob
from typing import List

import numpy as np
import pandas as pd

import assets as util
from assets import viewport as vp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.figure import Axes
from pathlib import Path

"""
Esta função já foi passada para o arquivo viewport.py e já pode ser apagada. Vou deixar aqui por enquanto só pra referência por causa dos
histogramas. (02/05/2021)
"""

def main():
    for idx, user_csv in enumerate(sorted(glob('data/*.csv'))):
        # if 'user03' not in user_csv: continue
        df_positions = pd.read_csv(user_csv, index_col=0)
        ax: List[Axes]
        fig: Figure
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        fig.suptitle(os.path.splitext(os.path.split(user_csv)[1])[0])
        ax[0].plot(df_positions.iloc[:, 0])
        ax[0].set_ylim(-180, 180)
        ax[1].plot(df_positions.iloc[:, 1])
        ax[1].set_ylim(-180, 180)
        ax[2].plot(df_positions.iloc[:, 2])
        ax[2].set_ylim(-180, 180)
        fig.show()
        pass

    print('finish')


def position2trajectory(user_csv='data/ride_user03_orientation.csv', fps=30):
    # (no.frames, cal.yaw, cal.pitch, cal.roll)
    user_csv = Path(user_csv)
    df_positions = pd.read_csv(user_csv, index_col=0)
    yaw_state = 0
    pitch_state = 0
    old_yaw = 0
    old_pitch = 0
    yaw_velocity = pd.Series(dtype=float)
    pitch_velocity = pd.Series(dtype=float)
    yaw_trajectory = []
    pitch_trajectory = []

    for frame, position in df_positions.iloc[:, 0:2].iterrows():
        """
        position: pd.Serie
        position.index == []
        """
        yaw = position.iloc[0]
        pitch = position.iloc[1]
        if not frame == 1:
            yaw_diff = yaw - old_yaw
            if yaw_diff > 200: yaw_state -= 1
            elif yaw_diff < -200: yaw_state += 1

            pitch_diff = pitch - old_pitch
            if pitch_diff > 120: pitch_state -= 1
            elif pitch_diff < -120: pitch_state += 1
            # print(f'Frame {n}, old={old:.3f}°, new={position:.3f}°, diff={diff :.3f}°')  # Want a log?

        new_yaw = yaw + 360 * yaw_state
        yaw_trajectory.append(new_yaw)

        new_pitch = pitch + 180 * pitch_state
        pitch_trajectory.append(new_pitch)

        if frame == 1:
            yaw_velocity.loc[frame] = 0
            pitch_velocity.loc[frame] = 0
        else:
            yaw_velocity.loc[frame] = (yaw_trajectory[-1] - yaw_trajectory[-2]) * fps
            pitch_velocity.loc[frame] = (pitch_trajectory[-1] - pitch_trajectory[-2]) * fps

        old_yaw = yaw
        old_pitch = pitch

    # Filter
    padded_yaw_velocity = [yaw_velocity.iloc[0]] + list(yaw_velocity) + [yaw_velocity.iloc[-1]]
    yaw_velocity_filtered = [sum(padded_yaw_velocity[idx - 1:idx + 2]) / 3
                             for idx in range(1, len(padded_yaw_velocity) - 1)]

    padded_pitch_velocity = [pitch_velocity.iloc[0]] + list(pitch_velocity) + [pitch_velocity.iloc[-1]]
    pitch_velocity_filtered = [sum(padded_pitch_velocity[idx - 1:idx + 2]) / 3
                               for idx in range(1, len(padded_pitch_velocity) - 1)]

    # Scalar velocity
    yaw_speed = np.abs(yaw_velocity_filtered)
    pitch_speed = np.abs(pitch_velocity_filtered)

    # Make graphs
    ax: List[List[plt.Axes]]
    fig: plt.Figure
    fig, ax = plt.subplots(3, 2, figsize=(16, 8), tight_layout=True)
    fig.suptitle(f'{user_csv}')

    ########
    ax[0][0].plot(df_positions.iloc[:, 0], label=f'Yaw. Avg={np.average(df_positions.iloc[:, 0]): .03f}')
    ax[0][0].set_ylim(-180, 180)
    ax[0][0].set_xlim(0, 1800)
    ax[0][0].legend()

    ax[0][1].plot(df_positions.iloc[:, 1], label=f'Pitch. Avg={np.average(df_positions.iloc[:, 1]): .03f}')
    ax[0][1].set_ylim(-90, 90)
    ax[0][1].set_xlim(0, 1800)
    ax[0][1].legend()

    ########
    ax[1][0].plot(yaw_trajectory, label=f'Trajectory of yaw. Avg={np.average(yaw_trajectory): .03f}')
    ax[1][0].axhline(180, color='gray', ls='--')
    ax[1][0].axhline(-180, color='gray', ls='--')
    ax[1][0].set_xlim(0, 1800)
    ax[1][0].legend()

    ax[1][1].plot(pitch_trajectory, label=f'Trajectory of pitch. Avg={np.average(df_positions.iloc[:, 1]): .03f}')
    ax[1][1].axhline(90, color='gray', ls='--')
    ax[1][1].axhline(-90, color='gray', ls='--')
    ax[1][1].set_xlim(0, 1800)
    ax[1][1].legend()

    ########
    ax[2][0].plot(yaw_velocity * 30)
    ax[2][0].plot(yaw_velocity_filtered, label=f'Avg spd = {np.average(yaw_velocity_filtered):.03f}°/s')
    ax[2][0].set_xlim(0, 1800)
    ax[2][0].set_xlabel('Frames')
    ax[2][0].legend()

    ax[2][1].plot(pitch_velocity * 30)
    ax[2][1].plot(pitch_velocity_filtered, label=f'Avg spd = {np.average(pitch_velocity_filtered):.03f}°/s')
    ax[2][1].set_xlim(0, 1800)
    ax[2][1].set_xlabel('Frames')
    ax[2][1].legend()

    ########
    # fig.show()

    folder_to_save = Path('results/graphs')
    fig_path_plot = folder_to_save / Path(user_csv.stem + '_plot')
    fig.savefig(fig_path_plot)

    ###############################################
    fig, ax = plt.subplots(3, 2, figsize=(16, 8), tight_layout=True)

    #####
    ax[0][0].hist(df_positions.iloc[:, 0], bins=36, density=True,
                  label=f'Position. Avg={np.average(df_positions.iloc[:, 0]): .03f}')
    ax[0][1].hist(df_positions.iloc[:, 1], bins=18, density=True,
                  label=f'Position. Avg={np.average(df_positions.iloc[:, 1]): .03f}')
    ax[0][0].set_xlim(-180, 180)
    ax[0][1].set_xlim(-90, 90)

    #####
    ax[1][0].hist(yaw_trajectory, bins=36, density=True, label='Yaw Trajectory')
    ax[1][1].hist(pitch_trajectory, bins=18, density=True, label='Pitch Trajectory')
    ax[1][0].set_xlim(-180, 180)
    ax[1][1].set_xlim(-90, 90)

    #####
    ax[2][0].hist(yaw_velocity_filtered, bins=36, density=True,
                  label=f'Avg spd = {np.average(yaw_velocity_filtered):.02f}°/s\n'
                        f'Avg scalar spd = {np.average(np.abs(yaw_velocity_filtered)):.02f}°/s\n'
                        f'Std = {np.std(yaw_velocity_filtered):.02f}°/s\n'
                        f'median = {np.median(yaw_velocity_filtered):.02f}°/s\n')
    ax[2][1].hist(pitch_velocity_filtered, bins=18, density=True,
                  label=f'Avg spd = {np.average(pitch_velocity_filtered):.02f}°/s\n'
                        f'Avg scalar spd = {np.average(np.abs(pitch_velocity_filtered)):.02f}°/s\n'
                        f'Std = {np.std(pitch_velocity_filtered):.02f}°/s\n'
                        f'median = {np.median(pitch_velocity_filtered):.02f}°/s')
    # ax[2][0].set_xlim(-180, 180)
    # ax[2][1].set_xlim(-90, 90)

    ax[0][0].legend()
    ax[0][1].legend()
    ax[1][0].legend()
    ax[1][1].legend()
    ax[2][0].legend()
    ax[2][1].legend()
    # ax[2].legend(loc='upper right')
    fig.suptitle(f'Histogram - {user_csv}')

    # fig.show()
    fig_path_plot = folder_to_save / Path(user_csv.stem + '_hist')
    fig.savefig(fig_path_plot)
    print('')


if __name__ == "__main__":
    # main()
    for datafile in glob('data/ride_user*'):
        position2trajectory(user_csv=datafile)
