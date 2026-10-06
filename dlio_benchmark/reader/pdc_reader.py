"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as np
import time

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import Profile, dft_ai
from dlio_benchmark.reader.reader_handler import FormatReader

from pdc_ctypes import get_client

dlp = Profile(MODULE_DATA_READER)

# Must match the generator
RECORD_DTYPE = np.uint8

# filename -> (num_samples, dim1, dim2): shared across all readers in a
# worker process, so the shape lookup happens once per file per worker.
_shape_cache = {}

import os

# PDC-backed reader, via ctypes. Selected with dataset.format: pdc.
# Requires pdc_ctypes.py on PYTHONPATH.
# Objects are samples first (num_samples, dim1, dim2)

def object_name(path):
    return os.path.basename(path)

"""
Reader for PDC objects. One region transfer per sample.
"""
class PDCReader(FormatReader):
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.dtype = RECORD_DTYPE
        get_client("dlio")

    """
    Per-file (dim1, dim2), derived from record_length (needs record_length_stdev == 0).

    Don't read the '<name>-dims' object: every PDC read lands in the client
    cache, and cached objects that no rank hints break the prefetch data exchange.
    """
    def _shape(self, filename):
        if "shape" not in _shape_cache:
            side = int(self._args.dimension)   # same value the generator uses
            _shape_cache["shape"] = (side, side)
        return _shape_cache["shape"]


    @dlp.log
    def open(self, filename):
        super().open(filename)

        print(f"file open: {filename}")
        start = time.time()

        result = get_client("dlio").obj(object_name(filename))

        end = time.time()
        elapsed = end - start
        print(f"[RANK {self._args.my_rank}] pdc_reader: open({object_name(filename)}) total time: {elapsed:.4f} seconds", flush = True)
        
        # Open the object only. Fetching data here would make the per-sample
        # timing below meaningless.
        return result

    @dlp.log
    def close(self, filename):
        super().close(filename)
    
    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        start = time.time()
        _ = self.open_file_map[filename]
        dim1, dim2 = self._shape(filename)
        image = get_client("dlio").read(
            object_name(filename), (sample_index, 0, 0), (1, dim1, dim2), self.dtype
        )
        image = image.reshape(dim1, dim2)

        if self._args.my_rank == 0:
            print(f"{object_name(filename)} idx={sample_index} "
                  f"shape={image.shape} min={image.min()} max={image.max()} "
                  f"mean={image.mean():.2f} nbytes={image.nbytes}", flush=True)

        end = time.time()
        elapsed = end - start
        print(f"[RANK {self._args.my_rank}] pdc_reader: get_sample({object_name(filename)}) total time: {elapsed:.4f} seconds", flush = True)

        dlp.update(image_size=image.nbytes)
        dft_ai.update(image_size=image.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
