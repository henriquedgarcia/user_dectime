import unittest
from configparser import ConfigParser

import assets as vp


class ViewportTestCase(unittest.TestCase):
    def setUp(self):
        project = "config"

        self.config = ConfigParser()
        self.config.read(f'{project}.ini')
        self.config = self.config['main']

        self.fov = vp.Dimension(int(self.config['fov_x']),
                                int(self.config['fov_y']))
        self.proj_res = vp.Dimension(int(self.config['p_res_x']),
                                     int(self.config['p_res_y']))

        self.viewport = vp.Viewport(self.fov,
                                    self.proj_res)

    # def test_rotation(self):
    #     position = vp.Scale(0,
    #                                 0,
    #                                 0)
    #     self.viewport.set_position(position)
    #
    #     self.assertIs(position,
    #                   self.viewport.position)
    #     viewport = self.viewport._rotate(position, self.viewport.viewport)
    def test_rot_matrix(self):
        pass

    def test_rotate(self):
        viewport = self.viewport
        new_position = vp.Point_bcs(yaw=0,
                                    pitch=0,
                                    roll=0)

        view = viewport._rotate(new_position,
                                viewport.default_view)
        self.assertIs()


if __name__ == '__main__':
    unittest.main()
