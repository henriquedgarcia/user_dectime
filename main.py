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
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import viewport.util as util
import viewport.viewport as vp


def main():
    """
    Videos must have less tan 99999 frames. (
    :return:
    """
    config_file = "config.ini"
    config = vp.Config(config_file)
    benchmark_config = json.load(open('config.json', 'r'))
    tile_list = benchmark_config['tile_list']
    video_list = ['clans']
    # mp_context = util.make_process(vp.worker, 1, config)

    for user in range(1, 51):
        # if user is 2: break
        for video in video_list:
            if video in 'clans':
                video = 'ride'

            csv = f'data/{video}_user{user:02}_orientation.csv'
            head_positions = pd.read_csv(csv, index_col=0)

            work_folder = f'{config.project}/viewport_tiles/'
            os.makedirs(work_folder, exist_ok=True)

            for tilling in tile_list:
                # if tilling not in '1x1': break

                m, n = list(map(int, tilling.split('x')))
                results = dict(video=[],
                               user=[],
                               tilling=[],
                               frame=[],
                               viewport_tiles=[])

                filename = f'{work_folder}/{video}_user{user}_{tilling}.csv'
                if os.path.isfile(filename):
                    print(f'viewport_tiles: '
                          f'O arquivo "{video}_user{user}.csv" existe. '
                          f'Pulando')
                    continue

                for frame, [_, row] in enumerate(head_positions.iterrows(), 1):
                    # if frame is 5: break
                    position = util.Point_bcs(float(row[config.yaw_column]),
                                              float(row[config.pitch_column]),
                                              float(row[config.roll_column]))

                    if config.unit in 'rad':
                        position.yaw = np.rad2deg(position.yaw)
                        position.pitch = np.rad2deg(position.pitch)
                        position.roll = np.rad2deg(position.roll)

                    # task_make_projection
                    # func = dict(func=vp.task_make_projection)
                    # args = dict(func=vp.task_make_projection,
                    #             position=position,
                    #             user=user,
                    #             frame=int(frame),
                    #             video=video)
                    # mp_context.task_queue.put((func, args))

                    # task_viewport_alpha
                    # func = dict(func=vp.task_viewport_alpha)
                    # args = dict(user=user,
                    #             frame=int(frame),
                    #             video=video)
                    # mp_context.task_queue.put((func, args))

                    args = dict(position=position,
                                user=user,
                                frame=frame,
                                video=video,
                                tilling=util.Dimension(m, n),
                                results=results)
                    func = task_viewport_tiles
                    # mp_context.task_queue.put((func, args))
                    func(config, **args)

                # while mp_context.task_queue.qsize() > 100:
                #     time.sleep(1)

                tiles_df = pd.DataFrame(results)
                tiles_df.to_csv(filename, index=False)

    # join_process(mp_context)
    print('Fim')


def join_process(mp_context):
    must_stop = False
    while not must_stop:
        time.sleep(1)
        status_list = list(map(lambda element: element.is_alive(),
                               mp_context.procs))
        alive_rate = status_list.count(True)
        must_stop = (alive_rate is 0
                     or mp_context.abort.is_set()
                     or (mp_context.task_queue.empty()
                         and mp_context.active_count.value is 0))
    mp_context.abort.set()
    list(map(lambda p: p.join(), mp_context.procs))


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

    notify = (f'main viewport_tiles:{video}, user_{user}, frame {frame}, '
              f'pattern {pattern}')
    print(notify)

    # Get viewport
    viewport = vp.Viewport(config.fov, config.proj_res)
    view = viewport.set_position(position)

    # Tiles dimension
    tiles = []
    tile_width = int(config.proj_res.x / tilling.x)
    tile_high = int(config.proj_res.y / tilling.y)
    tiles_iter = itertools.product(range(tilling.y), range(tilling.x))
    for idx, [tile_h, tile_v] in enumerate(tiles_iter, 1):
        border = vp.get_border(tile_v, tile_h, tile_width, tile_high)

        for (x, y) in border:
            point = util.unproject(util.Dimension(x, y), config.proj_res)
            if vp.is_viewport(point, view):
                tiles.append(idx)
                break

    results['video'].append(video)
    results['user'].append(user)
    results['tilling'].append(pattern)
    results['frame'].append(frame)
    results['viewport_tiles'].append(tiles)


class Container:
    pass


if __name__ == "__main__":
    main()
