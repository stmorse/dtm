"""
Loads a few lines from a comments file and prints them
"""

import json

import bz2                # for .bz2
import lzma               # for .xz
import zstandard as zstd  # for .zst


def open_compressed(file_path):
    if file_path.endswith('.bz2'):
        return bz2.BZ2File(file_path, 'rb')
    elif file_path.endswith('.xz'):
        return lzma.open(file_path, 'rb')
    elif file_path.endswith('.zst'):
        # For .zst, return a stream reader
        f = open(file_path, 'rb')  # Open file in binary mode
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        return dctx.stream_reader(f)
        # return dctx.read_to_iter(f)
    else:
        raise ValueError('Unsupported file extension.')

def read_sentences(file_path, chunk_size=10000):
    """
    Read JSON entries from a compressed file, extract the 'body' field,
    and yield chunks of size `chunk_size`.
    """
    buffer = []  # To store 'body' fields in chunks
    byte_buffer = b""  # For handling partial lines in `.zst` files

    with open_compressed(file_path) as f:
        # Iterate over the file
        for chunk in iter(lambda: f.read(8192), b""):  # Read file in binary chunks
            byte_buffer += chunk

            # Process each line in the byte buffer
            while b"\n" in byte_buffer:
                line, byte_buffer = byte_buffer.split(b"\n", 1)

                # Parse JSON and process the 'body' field
                entry = json.loads(line.decode("utf-8"))

                if 'body' not in entry or entry['author'] == '[deleted]':
                    continue

                # Truncate long 'body' fields
                body = entry['body']
                if len(body) > 2000:
                    body = body[:2000]

                # Add to the chunk buffer
                buffer.append(body)
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = []

        # Handle any remaining partial JSON line
        if byte_buffer:
            entry = json.loads(byte_buffer.decode("utf-8"))
            if 'body' in entry and entry['author'] != '[deleted]':
                body = entry['body']
                if len(body) > 2000:
                    body = body[:2000]
                buffer.append(body)

        # Yield any leftovers in the chunk buffer
        if buffer:
            yield buffer


if __name__ == '__main__':
    res = read_sentences(
        '/sciclone/data10/twford/reddit/reddit/comments/RC_2016-01.bz2',
        chunk_size=10
    )

    k = 0
    for r in res:
        print(r)
        k += 1
        if k > 3:
            break