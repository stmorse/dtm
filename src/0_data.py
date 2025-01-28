# import os
# import bz2
# import json
# import pickle
# import time

# if __name__=="__main__":

#     base_path = '/sciclone/data10/twford/reddit/reddit/comments'
#     save_path = '/sciclone/geograd/stmorse/reddit'

#     start_year = 2007
#     end_year = 2014
#     start_month = 1
#     end_month = 12
#     years = range(start_year, end_year+1)
#     months = range(start_month, end_month+1)
#     yrmo = [(yr, mo) for yr in years for mo in months]

#     counts = {}
#     t0 = time.time()
#     for year, month in yrmo:
#         path = os.path.join(base_path, f'RC_{year}-{month:02}.bz2')
#         c = 0
#         print(f'Counting {year}-{month} ... ({time.time()-t0:.3f})')
#         with bz2.BZ2File(path, 'rb') as f:
#             for line in f:
#                 entry = json.loads(line)
#                 if 'body' in entry and entry['author'] != '[deleted]':
#                     c += 1
#         counts[f'{year}-{month}'] = c
#         print(f'  Complete: {c} lines.')
    
#     with open(os.path.join(save_path, 'line_counts.pkl'), 'w') as f:
#         pickle.dump(counts, f)
    
#     print(f'Complete. ({time.time()-t0:.3f})')


import zstandard as zstd

def get_frame_info(file_path):
    with open(file_path, 'rb') as f:
        head = f.read(128)
    return zstd.get_frame_parameters(head)

file_path = "/sciclone/data10/twford/reddit/reddit/comments/RC_2020-02.zst"
frame_params = get_frame_info(file_path)
print(f"Window size: {frame_params.window_size}")
print(f"Content size: {frame_params.content_size}")
