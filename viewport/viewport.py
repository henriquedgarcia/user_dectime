import itertools
import multiprocessing as mp
import os

import cv2
import numpy as np

import viewport.util as util


class Config(util.ConfigBase):
    def __init__(self, config_file):
        self.config_file = config_file
        self.project = ''
        self.projection = ''
        self.p_res_x = 0
        self.p_res_y = 0
        self.fov_x = 0
        self.fov_y = 0
        self.unit = ''
        self.yaw_column = ''
        self.pitch_column = ''
        self.roll_column = ''
        self.fov = util.Dimension(0, 0)
        self.proj_res = util.Dimension(0, 0)
        self.pattern = util.Dimension(0, 0)

        self.configure(config_file)
        os.makedirs(self.project, exist_ok=True)

        self.pattern = util.Dimension(*list(map(int, self.pattern.split('x'))))
        self.fov = util.Dimension(self.fov_x, self.fov_y)
        self.proj_res = util.Dimension(self.p_res_x, self.p_res_y)


class Plane:
    normal: util.Point3d
    relation: str

    def __init__(self,
                 normal=util.Point3d(0, 0, 0),
                 relation='<'):
        self.normal = normal
        self.relation = relation


class View:
    def __init__(self, fov=util.Dimension(0, 0)):
        """
        The viewport is the a region of sphere created by the intersection of
        four planes that pass by center of sphere and the Field of view angles.
        Each plane split the sphere in two hemispheres and consider the viewport
        overlap.
        Each plane was make using a normal vectors (x1i+y1j+z1k) and the
        equation of the plane (x1x+y2y+z1z=0)
        If we rotate the vectors, so the viewport is roted too.
        :param fov: Field-of-View in degree
        :return: None
        """
        self.center = util.Point_hcs(0, 0, 0)
        self.fov = fov

        fovx = np.deg2rad(fov.x)
        fovy = np.deg2rad(fov.y)

        self.p1 = Plane(util.Point3d(-np.sin(fovy / 2), 0, np.cos(fovy / 2)))
        self.p2 = Plane(util.Point3d(-np.sin(fovy / 2), 0, -np.cos(fovy / 2)))
        self.p3 = Plane(util.Point3d(-np.sin(fovx / 2), np.cos(fovx / 2), 0))
        self.p4 = Plane(util.Point3d(-np.sin(fovx / 2), -np.cos(fovx / 2), 0))

    def __iter__(self):
        return iter([self.p1, self.p2, self.p3, self.p4])

    def get_planes(self):
        return [self.p1, self.p2, self.p3, self.p4]


class Viewport:
    def __init__(self, fov: util.Dimension, proj_res: util.Dimension) -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        :param proj_res:
        """
        self.position = util.Point_bcs(0, 0, 0)
        self.projection = np.array([])
        self.fov = fov
        self.proj_res = proj_res
        self.view = View(fov)
        self.new_view = View()
        self.pre_proj = np.zeros((self.proj_res.y, self.proj_res.x))

    def set_position(self, new_position: util.Point_bcs) -> View:
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Return numpy.ndarray.
        :param new_position:
        :return:
        """
        self.position = new_position
        self.new_view = rotate(new_position, self.view)
        return self.new_view

    def project(self) -> np.ndarray:
        self.projection = project_viewport(self.new_view, self.proj_res)
        return self.projection

    def show(self) -> None:
        cv2.imshow('imagem', self.projection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def rotate(new_position: util.Point_bcs, view: View):
    """
    Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
    angles in Z-Y-X order. Refer to Wikipedia.
    :param new_position: A point using Body Coordinate System.
    :param view: The View object that will be rotate.
    :return: A new View object with the new position.
    """
    new_view = View(view.fov)
    mat = util.rot_matrix(new_position)

    for plane1, plane2 in zip(view, new_view):
        normal = plane1.normal
        t3 = mat @ normal
        plane2.normal = util.Point3d(*t3)

    view.center = util.Point_hcs(1, new_position[0], new_position[1])
    return new_view


def project_viewport(view: View, res: util.Dimension) -> np.ndarray:
    """
    Project the sphere using ERP. Where is Viewport the
    :param view: A View object to project.
    :param res: The resolution of the Viewport
    :return: a numpy.ndarray with one deep color
    """
    projection = np.zeros((res.y, res.x), dtype=np.uint8)
    for j, i in itertools.product(range(res.y), range(res.x)):
        point3d = util.unproject(util.Point2d(i, j), res)
        if is_viewport(point3d, view):
            projection.itemset((j, i), 255)
    return projection


def is_viewport(point: util.Point3d, view: View) -> bool:
    """
    Check if the plane equation is true to viewport
    x1 * x + y1 * y + z1 * z < 0
    If True, the "point" is on the viewport
    :param point: A 3D Point in the space.
    :param view: A View object with four normal vector.
    :return: A boolean
    """
    is_in = True
    for plane in view.get_planes():
        result = (plane.normal.x * point.x
                  + plane.normal.y * point.y
                  + plane.normal.z * point.z)
        teste = (result < 0)

        # is_in só retorna true se todas as expressões forem verdadeiras
        is_in = is_in and teste
    return is_in


def worker(q: mp.Queue, active_count: mp.Value, abort: mp.Event,
           config: Config):
    from queue import Empty
    while not abort.is_set():
        try:
            item = q.get(timeout=1)
            with active_count.get_lock(): active_count.value += 1
        except Empty:
            continue

        # noinspection PyBroadException
        try:
            func, args = item['func']
            func(config, **args)
        except Exception:
            print(f'O Worker {mp.current_process().name} parou')
            abort.set()

        with active_count.get_lock(): active_count.value -= 1


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
    viewport = Viewport(fov=config.fov, proj_res=config.proj_res)
    viewport.set_position(position)
    projection = viewport.project()

    cv2.imwrite(filename, projection)


def task_viewport_alpha(config, user, frame, video):
    # make projection as alpha channel of video frame
    p_name = mp.current_process().name
    work_folder = f'{config.project}/viewport_alpha'
    os.makedirs(work_folder, exist_ok=True)
    filename = f'{work_folder}/{video}_user{user}_{frame:05}.png'

    if os.path.isfile(filename):
        print(f'viewport_alpha: '
              f'O arquivo {video}_user{user}_{frame:05}.png existe. Pulando')
        return

    notify = f'{p_name} viewport_alpha:{video}, user_{user}, frame {frame}'
    print(notify)

    # working
    projection_folder = f'{config.project}/viewport_projection/'
    projection_filename = (f'{projection_folder}/'
                           f'{video}_user{user}_{frame:05}.png')
    projection = cv2.imread(projection_filename)

    frame_name = f'frames/{video}{frame:04}.png'
    frame = cv2.imread(frame_name, cv2.IMREAD_UNCHANGED)
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

    pattern = f'{tilling.x}x{tilling.y}'

    # notify = (f'main viewport_tiles:{video}, user_{user}, frame {frame}, '
    #           f'pattern {pattern}')
    # print(notify)

    # Get viewport
    viewport = Viewport(config.fov, config.proj_res)
    view = viewport.set_position(position)

    # Tiles dimension
    tiles = []
    tile_width = int(config.proj_res.x / tilling.x)
    tile_high = int(config.proj_res.y / tilling.y)
    tiles_iter = itertools.product(range(tilling.y), range(tilling.x))
    for idx, [tile_h, tile_v] in enumerate(tiles_iter, 1):
        border = get_border(tile_v, tile_h, tile_width, tile_high)

        for (x, y) in border:
            point = util.unproject(util.Dimension(x, y), config.proj_res)
            if is_viewport(point, view):
                tiles.append(idx)
                break

    results['video'].append(video)
    results['user'].append(user)
    results['tilling'].append(pattern)
    results['frame'].append(frame)
    results['viewport_tiles'].append(tiles)


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
