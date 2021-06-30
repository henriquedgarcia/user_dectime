import json
import math
import multiprocessing as mp
import subprocess
from typing import Any, Dict, List, NamedTuple

import numpy as np
import pandas as pd

Dimension = NamedTuple('Dimension',
                       [('x', int), ('y', int)])

Point3d = NamedTuple('Point3d',
                     [('x', float), ('y', float), ('z', float)])

Point2d = NamedTuple('Point2d',
                     [('x', float), ('y', float)])

# body coordinate system
Point_bcs = NamedTuple('Point_bcs',
                       [('yaw', float), ('pitch', float), ('roll', float)])

# horizontal coordinate system
Point_hcs = NamedTuple('Point_hcs',
                       [('r', float), ('azimuth', float), ('elevation', float)])

# Multiprocessing object
ProcContext = NamedTuple('ProcContext',
                         [('procs', List),
                          ('task_queue', mp.Queue),
                          ('active_count', mp.Value),
                          ('abort', mp.Event)])


class ConfigBase:
    """
    Base class to create config object.
    """
    config_file: str

    def configure(self, config: Dict[str, Any]) -> None:
        """
        This function convert itens under [main] sections of a config file
        created using ConfigParser in attributes of this class.
        :param config: the config filename
        :return: None
        """
        for item in config:
            try:
                value = float(config[item])
            except ValueError:
                value = config[item]
            setattr(self, item, value)


def hcs2cart(position: Point_hcs):
    """
    Horizontal Coordinate system to Cartesian coordinates
    :param position: The coordinates in Horizontal Coordinate System
    :return: A Point3d in cartesian coordinates
    """
    az = position.azimuth/180 * math.pi
    el = position.elevation/180 * math.pi
    r = position.r

    ca = math.cos(az)
    ce = math.cos(el)
    sa = math.sin(az)
    se = math.sin(el)
    p = Point3d(r * ce * ca,
                r * ce * sa,
                r * se)
    return p


def cart2hcs(position: Point3d) -> Point_hcs:
    """
    Cartesian coordinates to Horizontal Coordinate system.
    :param position: The cartesian coordinates to convert.
    :return: a Point_hcs object with angles in degree
    """
    # r = math.sqrt(position.x ** 2 + position.y ** 2 + position.z ** 2)
    # p = Point_hcs(r,
    p = Point_hcs(1,
                  np.rad2deg(math.atan2(position.y, position.x)),
                  np.rad2deg(math.atan2(position.z,
                                        np.sqrt(position.x ** 2 + position.y ** 2))))

    return p


def rot_matrix(new_position: Point_bcs):
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia.
    :param new_position: A new position using the Body Coordinate System.
    :return:
    """
    yaw = np.deg2rad(new_position.yaw)
    pitch = np.deg2rad(new_position.pitch)
    roll = np.deg2rad(new_position.roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)

    mat = np.array(
        [[cy * cp, cy * sp * sr - cr * sy, sy * sr + cy * cr * sp],
         [cp * sy, cy * cr + sy * sp * sr, cr * sy * sp - cy * sr],
         [-sp, cp * sr, cp * cr]])

    return mat


def unproject(point: Point2d, res: Dimension) -> Point3d:
    """
    Convert a 2D point of ERP projection coordinates to Horizontal Coordinate
    System
    :param point: A point in ERP projection
    :param res: The resolution of the projection
    :return: A 3D Point on the sphere
    """

    # Only ERP Projection
    point = hcs2cart(Point_hcs(1,
                               (point.x / res.x) * 360 - 180,
                               (point.y / res.y) * 180 - 90))
    return point


def make_process(func, proc_num=1, *args):
    """
    Make multiprocess of the function and pass the args values.
    :param func: a function that receive shared queue, shared count and *args
    :param proc_num: Process numbers. 0 is auto
    :param args: extra args passed to process
    :return: a dict containing a processes list, a FIFO shared queue and a
    shared counter of active process.
    """
    if proc_num < 1:
        proc_num = max(4, mp.cpu_count())
        print(f'creating {proc_num} process')

    task_queue = mp.Queue()
    active_count = mp.Value('i', 0)
    abort = mp.Event()

    procs = []
    for _ in range(proc_num):
        p = mp.Process(target=func, args=(task_queue, active_count, abort, *args))
        p.start()
        procs.append(p)

    mp_context = ProcContext(procs=procs, task_queue=task_queue,
                             active_count=active_count, abort=abort)

    return mp_context


class ProxyDataFrame:
    """
    Classe usada como multiprocessing.proxy que coordena a inserção de linhas em
    um pandas.DataFrame compartilhado por um multiprocessing.Manager.
    """
    def __init__(self, cols):
        self.cols = cols
        self.data_frame = pd.DataFrame(columns=cols)

    def add(self, obj):
        """
        Adiciona uma lista ao dataframe membro
        Args:
            obj (list): Uma lista a ser adicionada. Deve ter o mesmo formado de
            sua definição.
        Return:
            None
        """
        # Adicione a lista no final do dataframe
        df = pd.DataFrame([obj], columns=self.cols)
        try:
            self.data_frame = self.data_frame.append(df)
        except Exception:
            msg = 'ProxyDataFrame: Erro ao adicionar lista ao dataframe.'
            print(msg)

    def copy(self):
        """
        Função que retorna uma cópia do Dataframe membro
        Return:
             (DataFrame): Uma cópia do DataFrame membro
        """
        return self.data_frame.copy()


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def run_command(command: str, log_to_save: str, mode: str = 'w') -> str:
    """
    Run a shell command with subprocess module with realtime output.
    :param command: A command string to run.
    :param log_to_save: A path-like to save the process output.
    :param mode: The write mode: 'w' or 'a'.
    :return: stdout.
    """
    print(command)

    with open(log_to_save, mode, encoding='utf-8') as f:
        process = subprocess.run(command, shell=True, stdout=f,
                                 stderr=subprocess.STDOUT, encoding='utf-8')
    return process.stdout


def save_json(data: dict, filename, compact=False):
    if compact:
        separators = (',', ':')
        indent = None
    else:
        separators = None
        indent = 2

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=separators, indent=indent)


def splitx(string: str) -> tuple:
    return tuple(map(int, string.split('x')))


def autocorrelation(array: list, positive_only=False):
    autocorrelate = np.correlate(array, array, mode="full")
    if positive_only:
        autocorrelate = autocorrelate[autocorrelate.size // 2:]
    return autocorrelate


def position2trajectory(positions: list, angles='euler'):
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