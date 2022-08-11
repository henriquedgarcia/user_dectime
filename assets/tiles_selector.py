import itertools
import multiprocessing as mp
import os
from itertools import product
from logging import warning
from typing import Union, List

import cv2
import numpy as np
import pandas as pd

from assets.viewport import Viewport, Dimension, View, Point_hcs, Point_bcs, Point3d, Point2d, rot_matrix, proj2sph, \
    hcs2cart, Config


class TilesSelector:
    viewport: Viewport


def task_make_projection(config, position, user, frame, video):
    # Process projection
    p_name = mp.current_process().name
    work_folder = f'{config.project}/viewport_projection/user_{user}'
    os.makedirs(work_folder, exist_ok=True)
    filename = f'{work_folder}/{video}_user{user}_{frame:05}.png'

    if os.path.isfile(filename):
        print(f'make_projection: '
              f'O arquivo {video}_user{user}_{frame:05}.png existe. Pulando')
        return

    notify = f'{p_name} make_projection:{video}, user_{user}, frame {frame}'
    print(notify)

    # working
    viewport = Viewport(fov=config.fov, scale=config.proj_res)
    viewport.set_position(position)
    projection = viewport.project()

    cv2.imwrite(filename, projection)


def task_viewport_alpha(projection, workfolder, basename, frame_num):
    # make projection as alpha channel of video frame
    p_name = mp.current_process().name
    work_folder = f'{workfolder}/viewport_alpha'
    os.makedirs(work_folder, exist_ok=True)
    os.makedirs(f'{work_folder}/frames/', exist_ok=True)
    filename = f'{work_folder}/frames/{basename}_{frame_num:05}.png'

    if os.path.isfile(filename):
        warning(f'O arquivo {basename}_{frame_num:05}.png existe. Pulando')
        return

    notify = f'{p_name} viewport_alpha:{basename}, frame {frame_num}'
    print(notify)

    frame = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    frame = cv2.resize(frame, dsize=projection.size(),
                       interpolation=cv2.INTER_NEAREST)

    result = np.dstack((frame, projection))

    cv2.imwrite(filename, result)

    # cv2.imshow('a',v_frame);cv2.waitKey();cv2.destroyAllWindows()
    # cv2.imshow('a',projection);cv2.waitKey();cv2.destroyAllWindows()


def task_viewport_tiles(config, user, frame, video, position, tilling,
                        results):
    """
    Make projection as alpha channel of video frame
    :param config:
    :param user:
    :param frame:
    :param video:
    :param video:
    :param position:
    :param tilling:
    :param results:
    :return:
    """
    # p_name = mp.current_process().name

    pattern = f'{tilling.m}m{tilling.n}'

    # notify = (f'main viewport_tiles:{video}, user_{user}, frame {frame}, '
    #           f'pattern {pattern}')
    # print(notify)

    # Get viewport
    viewport = Viewport(config.fov)
    view = viewport.set_position(position)

    # Tiles dimension
    tiles = []
    tile_width = int(config.proj_res.m / tilling.m)
    tile_high = int(config.proj_res.n / tilling.n)
    tiles_iter = itertools.product(range(tilling.n), range(tilling.m))
    for idx, [tile_h, tile_v] in enumerate(tiles_iter, 1):
        border = get_border(tile_v, tile_h, tile_width, tile_high)

        for (x, y) in border:
            point = proj2sph(Dimension(x, y), config.proj_res)
            if is_viewport(point, view):
                tiles.append(idx)
                break

    results['video'].append(video)
    results['user'].append(user)
    results['tilling'].append(pattern)
    results['frame'].append(frame)
    results['viewport_tiles'].append(tiles)

def rotate(view: View, new_position: Point_bcs):
    """
    Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
    angles in Z-Y-X order. Refer to Wikipedia.
    :param new_position: A point using Body Coordinate System.
    :param view: The View object that will be rotate.
    :return: A new View object with the new position.
    """
    new_view = View(view.fov)
    mat = rot_matrix(new_position)

    # For each plane in view
    for default_plane, new_plane in zip(view, new_view):
        normal = default_plane.normal
        roted_normal = mat @ normal
        new_plane.normal = Point3d(roted_normal[0], roted_normal[1], roted_normal[2])

    view.center = Point_hcs(1, new_position[0], new_position[1])
    return new_view

def project_viewport(view: View, res: Dimension) -> np.ndarray:
    """
    Project the sphere using ERP. Where is Viewport the
    :param view: A View object to project.
    :param res: The resolution of the Viewport
    :return: a numpy.ndarray with one deep color
    """
    projection = np.ones((res.n, res.m), dtype=np.uint8) * 255
    for j, i in product(range(res.n), range(res.m)):
        point_hcs = proj2sph(Point2d(i, j), res)
        point_cart = hcs2cart(point_hcs)

        if is_viewport(point_cart, view):
            projection.itemset((j, i), 0)  # by the docs, it is more efficient than projection[j, i] = 0
    return projection

def is_viewport(point: Point3d, view: View) -> bool:
    """
    Check if the plane equation is true to viewport
    x1 * m + y1 * n + z1 * z < 0
    If True, the "point" is on the viewport
    :param point: A 3D Point in the space.
    :param view: A View object with four normal vector.
    :return: A boolean
    """
    is_in = True
    for plane in view.get_planes():
        result = (plane.normal.m * point.x
                  + plane.normal.n * point.y
                  + plane.normal.z * point.z)
        teste = (result < 0)

        # is_in só retorna true se todas as expressões forem verdadeiras
        is_in = is_in and teste
    return is_in


def get_border(x: int, y: int, width: int, high: int) -> list:
    """
    :param x: coordenada horizontal do tile em pixels
    :param y: coordenada vertical do tile em pixels
    :param width: largura do tile em pixels
    :param high: altura do tile em pixels
    :return: list with all border pixels coordinates
    """
    # Limits
    x1 = range(width * x, width + width * x)  # correndo a linha
    x2 = [width * x]  # parado na primeira linha
    x3 = [width * (x + 1) - 1]  # parado na ultima linha
    y1 = range(high * y, high * y + high)  # correndo a coluna
    y2 = [0 + high * y]  # parado na primeira coluna
    y3 = [high * (y + 1) - 1]  # parado na ultima coluna

    # Borders
    border = list(zip(x1, y2 * width))
    border.extend(list(zip(x2 * high, y1)))
    border.extend(list(zip(x1, y3 * width)))
    border.extend(list(zip(x3 * high, y1)))

    return border


def position2trajectory(positions: Union[pd.Series, List], rng=360, fps=30, threshold=0.75):
    """
    positions: Iterable of int or float. Values must be degree.
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


def averaging_filter(serie: list, padding_type='repeat values'):
    if 'repeat values' in padding_type:
        padded_serie = serie[0] + serie + serie[-1]
    elif 'zeros' in padding_type:
        padded_serie = [0] + serie + [0]
    else:
        raise ValueError(f'padding_type "{padding_type}" not supported.')

    serie_filtered = [(padded_serie[idx - 1] + padded_serie[idx] + padded_serie[idx + 1]) / 3
                      for idx in range(1, len(padded_serie) - 1)]
    return serie_filtered


def worker(q: mp.Queue, active_count: mp.Value, abort: mp.Event,
           config: Config):
    from queue import Empty
    while not abort.is_set():
        try:
            item = q.get(timeout=1)
            with active_count.get_lock():
                active_count.value += 1
        except Empty:
            continue

        # noinspection PyBroadException
        try:
            func, args = item['func']
            func(config, **args)
        except Exception:
            print(f'O Worker {mp.current_process().name} parou')
            abort.set()

        with active_count.get_lock():
            active_count.value -= 1


def position2trajectory_util(positions: list, angles='euler'):
    state = 0
    old = 0
    derived = []
    trajectory = []
    for n, position in enumerate(positions):
        if n == 0:
            old = position
            continue
        diff = old - position
        derived.append(diff)

        if diff > 200: state += 1
        elif diff < -200: state -= 1

        new_position = position + 360*state
        trajectory.append(new_position)
        old = position
    return trajectory