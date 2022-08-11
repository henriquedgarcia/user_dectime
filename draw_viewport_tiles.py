import numpy as np
import pandas as pd

from assets import viewport as vp


def main():
    config = vp.Config("config.ini")
    viewport = vp.Viewport(config.fov)
    head_positions = pd.DataFrame({'yaw': np.arange(0, 360, 30),
                                   'pitch': [0] * len(np.arange(0, 360, 30)),
                                   'roll': [0] * len(np.arange(0, 360, 30))})

    for frame_idx, row in head_positions.iterrows():
        viewport.set_position(vp.Point_bcs(*row))
        viewport.project('400x200')
        viewport.show()
        viewport.save('viewport.png')
        print('')


if __name__ == '__main__':
    main()
