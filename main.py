#!/usr/bin/python3
"""
para cada usuário
    leia o csv
    faça o viewport da projeção
        - criar o viewport para o ponto 0,0,0 no domínio da esfera.
        para cada quadro
            - Rodar o viewport de acordo com o os dados do CSV
            - Projetar o viewport na projeção desejada.
            - salvar
    determine os tiles do viewport
        para cada quadro:
        - segmentar a projeção de acordo com o padrão de tiles considerado.
        - se algum pixel do viewport estiver no tile ele é selecionado.
        - 0 para um tile que não pixel e 1 para o que tem:
            ex, segmentação 4x4:
            a = [[0, 0, 0, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 0]], type=bool
            considerar todos os quadros
        - o resultado para cada video e cada padrão e cada qualidade:
        um vetor de matrizes
    faça a soma da decodificação por chunk time (1s)
        abrir dados de decodificação
"""

from assets import viewport as vp
import matplotlib.pyplot as plt
tile_list = ["1x1", "3x2", "4x3", "6x3", "6x4", "6x5", "6x6", "7x6"]
video_list = ['clans']


def main():
    position = vp.Point_bcs(0, 0, 0)
    tiling = vp.Tiling('8x6', '400x200', '120x90')
    vptiles = tiling.get_vptiles(position)
    print('Fim')
    print(f'{vptiles}')


# def task_viewport_tiles(config, user, frame, video, position, tilling,
#                         results) -> pd.DataFrame:
#     """
#     Make projection as alpha channel of video frame
#     :param config:
#     :param user:
#     :param frame:
#     :param video:
#     :param video:
#     :param position:
#     :param tilling:
#     :param results:
#     :return:
#     """
#     # p_name = mp.current_process().name
#
#     pattern = f'{tilling.m}x{tilling.n}'
#
#     notify = (f'main viewport_tiles:{video}, user_{user}, frame {frame}, '
#               f'pattern {pattern}')
#     print(notify)
#
#     # Get viewport
#     viewport = vp.Viewport(config.viewport.fov, config.scale)
#     view = viewport.set_position(position)
#
#     # Tiles dimension
#     tiles = []
#     tile_width = int(config.proj_res.m / tilling.m)
#     tile_high = int(config.proj_res.n / tilling.n)
#     tiles_iter = itertools.product(range(tilling.n), range(tilling.m))
#     idx = 0
#     for tile_y in range(tilling.n):
#         for tile_x in range(tilling.m):
#             idx += 1
#
#             border = vp.get_border(tile_x, tile_y)
#
#             for (x, y) in border:
#                 point = util.unproject(util.Dimension(x, y), config.proj_res)
#                 if vp.is_viewport(point, view):
#                     tiles.append(idx)
#                     break
#
#     results['video'].append(video)
#     results['user'].append(user)
#     results['tilling'].append(pattern)
#     results['frame'].append(frame)
#     results['viewport_tiles'].append(tiles)
#     return results


if __name__ == "__main__":
    main()
