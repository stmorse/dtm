import os
import json
import bz2
import lzma
import zstandard as zstd

# Directories from your original script:
COMMENTS_DIR = "/sciclone/data10/twford/reddit/reddit/comments"
SUBMISSIONS_DIR = "/sciclone/data10/twford/reddit/reddit/submissions"

# Months to check from January 2006 to January 2023:
MONTHS = [f"{year}-01" for year in range(2016, 2017)]

def open_compressed_file(fname):
    if fname.endswith(".bz2"):
        return bz2.open(fname, "rt", encoding="utf-8")
    elif fname.endswith(".xz"):
        return lzma.open(fname, "rt", encoding="utf-8")
    elif fname.endswith(".zst"):
        import io
        f = open(fname, "rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        return io.TextIOWrapper(dctx.stream_reader(f), encoding='utf-8')
    else:
        raise ValueError(f"Unsupported extension for {fname}")

def get_first_row(filepath):
    try:
        with open_compressed_file(filepath) as f:
            for line in f:
                data = json.loads(line)
                return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def find_file(directory, prefixes, month):
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefixes) and month in filename:
            return os.path.join(directory, filename)
    return None

def check_files_for_month(month):
    print(f"\n========== Checking Data for {month} ==========")
    
    print("\n--- Submissions ---")
    submissions_file = find_file(SUBMISSIONS_DIR, ('RS_', 'RS_v2_'), month)
    if submissions_file:
        first_row = get_first_row(submissions_file)
        if first_row:
            print(f"File: {submissions_file}")
            print("Variables (Keys):")
            print(list(first_row.keys()))
            print("First Row:")
            print(first_row)
    else:
        print(f"No submissions file found for {month}.")

    print("\n--- Comments ---")
    comments_file = find_file(COMMENTS_DIR, 'RC_', month)
    if comments_file:
        first_row = get_first_row(comments_file)
        if first_row:
            print(f"File: {comments_file}")
            print("Variables (Keys):")
            print(list(first_row.keys()))
            print("First Row:")
            print(first_row)
    else:
        print(f"No comments file found for {month}.")

def main():
    for month in MONTHS:
        check_files_for_month(month)

if __name__ == "__main__":
    main()