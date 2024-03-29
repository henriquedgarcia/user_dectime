import itertools
import multiprocessing as mp
import os

import cv2
import numpy as np

import assets.util as util


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

        self.pattern = util.Dimension(*list(map(int, self.pattern.split('m'))))
        self.fov = util.Dimension(self.fov_x, self.fov_y)
        self.proj_res = util.Dimension(self.p_res_x, self.p_res_y)


class Viewport:
    def __init__(self, fov: util.Dimension, proj_res: util.Dimension) -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        :param proj_res:
        """
        self.position = util.Point_bcs(0, 0, 0)
        self.new_position = util.Point_bcs(0, 0, 0)
        self.projection = np.array([])

        self.fov = fov
        self.proj_res = proj_res
        self.view = View(fov)
        self.new_view = View()

    def set_position(self, new_position: util.Point_bcs) -> np.ndarray:
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Return numpy.ndarray.
        :param new_position:
        :return:
        """
        self.new_position = new_position
        self.new_view = rotate(new_position, self.view)
        self.projection = project_viewport(self.new_view, self.proj_res)
        return self.projection

    def show(self) -> None:
        cv2.imshow('imagem', self.projection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
    x1 * m + y1 * n + z1 * z < 0
    If True, the "point" is on the viewport
    :param point: A 3D Point in the space.
    :param view: A View object with four normal vector.
    :return: A boolean
    """
    is_in = True
    for i, plane in enumerate(view):
        result = (plane.normal.m * point.x
                  + plane.normal.n * point.y
                  + plane.normal.z * point.z)
        teste = (result < 0)

        # is_in só retorna true se todas as expressões forem verdadeiras
        is_in = is_in and teste

    return is_in


def worker(queue: mp.Queue, active_count: mp.Value, config: Config):
    try:
        viewport = Viewport(fov=config.fov, proj_res=config.proj_res)
        p_name = mp.current_process().name
        while True:
            item = queue.get()
            with active_count.get_lock():
                active_count.value += 1

            # split items
            position = item['position']
            user = item['user']
            frame_idx = item['frame']
            video = item['video']

            print(f'{p_name} - '
                  f'video: {video}, '
                  f'user: {user}, '
                  f'frame: {frame_idx}', flush=True)

            # Process projection
            projection = viewport.set_position(position)
            mask_filename = f'{config.project}/user{user}_{frame_idx:05}.png'
            cv2.imwrite(mask_filename, projection)

            # make projection as alpha channel of video frame
            frame_name = f'frames/{video}{frame_idx:04}.png'
            v_frame = cv2.imread(frame_name, cv2.IMREAD_UNCHANGED)
            result = np.dstack((v_frame, projection))
            result_filename = f'{config.project}/{video}_user{user}_{frame_idx:05}.png'
            cv2.imwrite(result_filename, result)

            with active_count.get_lock():
                active_count.value -= 1

            # cv2.imshow('a',v_frame);cv2.waitKey();cv2.destroyAllWindows()
            # viewport.show()

    except Exception as e:
        print('O Worker XXX parou.')
        raise e
