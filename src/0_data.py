import io
import os
import zipfile
import numpy as np
import zarr

# def count_npz_rows(npz_path):
#     # total = 0
#     # with zipfile.ZipFile(npz_path, 'r') as z:
#     #     for fname in z.namelist():
#     #         with z.open(fname) as f:
#     #             # Read the magic string then header.
#     #             magic = np.lib.format.read_magic(f)
#     #             # Warning: _read_array_header is internal API.
#     #             header, _, _ = np.lib.format._read_array_header(f, magic)
#     #             total += header['shape'][0]
#     with open(npz_path, 'rb') as f:
#         arr = np.load(f)['embeddings']
#         return arr.shape[0]
#     # return total



def count_npz_rows(npz_path):
    total = 0
    with zipfile.ZipFile(npz_path, 'r') as z:
        for name in z.namelist():
            if not name.endswith('.npy'):
                continue
            with z.open(name) as f:
                bf = io.BufferedReader(f)
                magic = np.lib.format.read_magic(bf)
                if magic == (1, 0):
                    res = np.lib.format.read_array_header_1_0(bf)
                elif magic == (2, 0):
                    res = np.lib.format.read_array_header_2_0(bf)
                else:
                    raise ValueError("Unsupported .npy version: " + str(magic))
                if len(res[0]) == 2:
                    total += res[0][0]
                else:
                    continue
    return total

def count_zarr_rows(zarr_path):
    arr = zarr.open(zarr_path, mode='r')
    return arr.shape[0]

if __name__=="__main__":
    base_path = '/sciclone/geograd/stmorse/reddit/embeddings'

    years = range(2018, 2023)
    months = [f'{mo:02}' for mo in range(1,13)]
    yrmo = [(yr, mo) for yr in years for mo in months]

    total = 0
    for year, month in yrmo:
        # path to compressed file
        file_path = None
        extension = None
        for ext in ['.zarr', '.npz']:
            potential_path = os.path.join(base_path, f'embeddings_{year}-{month}{ext}')
            if os.path.exists(potential_path):
                file_path = potential_path
                extension = ext
                break
        if file_path is None:
            raise FileNotFoundError(
                f"No file found for {year}-{month} with supported extensions")
        
        subtotal = 0
        if extension == '.npz':
            subtotal = count_npz_rows(file_path)
        elif extension == '.zarr':
            subtotal = count_zarr_rows(file_path)
        print(f'{year}-{month}: {subtotal}')

        total += subtotal

    print(f'GRAND TOTAL: {total}')



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




# import zstandard as zstd

# def get_frame_info(file_path):
#     with open(file_path, 'rb') as f:
#         head = f.read(128)
#     return zstd.get_frame_parameters(head)

# file_path = "/sciclone/data10/twford/reddit/reddit/comments/RC_2020-02.zst"
# frame_params = get_frame_info(file_path)
# print(f"Window size: {frame_params.window_size}")
# print(f"Content size: {frame_params.content_size}")
