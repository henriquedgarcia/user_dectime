import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    """
    CSV must be like this:
    no.frames,cal.yaw,cal.pitch,cal.roll
    00001,5.45393630573,6.1422076287,-3.749
    00002,5.45393630573,6.1422076287,-3.749
    00003,5.45193630573,6.0572076287,-3.72
    00004,5.45193630573,6.0572076287,-3.72
    00005,5.46793630573,6.0212076287,-3.672
    """
    path = Path('D:/workbench/Projeto viewport Quality Streaming/user_dectime/data/_teste.csv')
    positions = pd.read_csv(path, index_col=0)
    yaw = positions.iloc[:, 0]
    yaw_traj, yaw_vel = position2trajectory(yaw)
    make_graph(yaw, yaw_traj, yaw_vel)


def make_graph(position, trajectory, velocity):
    fig, ax = plt.subplots(2, 1, figsize=(7, 5))
    ax[0].plot(position)
    ax[0].set_title('Position.')
    ax[0].set_ylabel('Deg.')

    ax[1].plot(trajectory)
    ax[1].set_title('Trajectory.')
    ax[1].set_xlabel('Frames')
    ax[1].set_ylabel('Deg.')
    fig.tight_layout()
    fig.show()

    #####
    fig, ax = plt.subplots(2, 1, figsize=(7, 5))
    ax[0].plot(velocity, label='Instantaneous velocity')
    ax[0].plot(averaging_filter(velocity), label='Average velocity')
    ax[0].set_title('Velocity.')
    ax[0].set_ylabel('Deg./s')
    ax[0].legend(loc='upper right')

    ax[1].plot(list(map(abs, velocity)), label='Instantaneous speed')
    ax[1].plot(averaging_filter(list(map(abs, velocity))), label='Average speed')
    ax[1].set_title('Speed.')
    ax[1].set_xlabel('Frames')
    ax[1].set_ylabel('Deg./s')
    ax[1].legend(loc='upper right')
    fig.tight_layout()
    fig.show()


def position2trajectory(positions: pd.Series, rng=360, fps=30, threshold=0.75):
    """
    positions: Must have frame no. as index. Values must be degree.
    rng: Range of dimension. 360 for Yaw, 180 for pitch.
    Return: trajectory and velocity
    """
    state = 0
    old_position = 0
    velocity = []
    trajectory = []

    for frame, position in enumerate(positions):
        if frame != 0:
            diff = position - old_position
            if diff > rng * threshold:
                state -= 1
            elif diff < -rng * threshold:
                state += 1

        trajectory.append(position + rng * state)
        old_position = position

        if frame == 0:
            velocity.append(0.)
        else:
            velocity.append((trajectory[-1] - trajectory[-2]) * fps)

    return trajectory, velocity


def averaging_filter(serie: list):
    padded_serie = [serie[0]] + serie + [serie[-1]]
    serie_filtered = [(padded_serie[idx - 1] + padded_serie[idx] + padded_serie[idx + 1]) / 3
                      for idx in range(1, len(padded_serie) - 1)]
    return serie_filtered


if __name__ == '__main__':
    main()
