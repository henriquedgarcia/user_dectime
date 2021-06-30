import itertools
import json
import pandas as pd
from assets import viewport as vp

if __name__ == '__main__':
    config_file = "config.ini"
    config = vp.Config(config_file)
    benchmark_config = json.load(open('config/config.json', 'r'))
    tile_list = benchmark_config['tile_list']

    for user in range(1, 51):
        # csv = f'data/ride_user{user:02}_orientation.csv'
        csv = f'data/ride_user01_orientation_teste.csv'
        head_positions = pd.read_csv(csv, index_col=0)

        for tilling in tile_list:
            used_tiles = pd.read_csv(f'{"clans"}_{user}.csv')
            m, n = list(map(int, tilling.split('x')))
            for frame_idx, [_, row] in enumerate(head_positions.iterrows(), 1):

                # A task começa aqui
                proj_res = config.proj_res
                print(f'user: {user}, '
                      f'tilling: {tilling}, '
                      f'frame: {frame_idx}', flush=True)
                tile_width = int(proj_res.x / tilling.x)
                tile_high = int(proj_res.y / tilling.y)
                tiles_iter = itertools.product(range(tilling.y),
                                               range(tilling.x))
                for idx, [tile_h, tile_v] in enumerate(tiles_iter, 1):
                    border = vp.get_border(tile_v, tile_h, tile_width,
                                           tile_high)

