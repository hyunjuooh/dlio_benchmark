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
import os
import time

import h5py

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import Profile, dft_ai
from dlio_benchmark.reader.reader_handler import FormatReader

dlp = Profile(MODULE_DATA_READER)

class HDF5Reader(FormatReader):
    """
    Reader for HDF5 files.
    """
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.dataset_indices = list(range(self._args.num_dset_per_record))

    @dlp.log
    def open(self, filename):
        super().open(filename)
        start = time.time()
        result = h5py.File(filename, 'r')
        elapsed = time.time() - start
        print(f"[RANK {self._args.my_rank}] hdf5_reader: open({os.path.basename(filename)}) total time: {elapsed:.4f} seconds", flush = True)
        return result

    @dlp.log
    def close(self, filename):
        start = time.time()
        self.open_file_map[filename].close()
        elapsed = time.time() - start
        print(f"[RANK {self._args.my_rank}] hdf5_reader: close({os.path.basename(filename)}) total time: {elapsed:.4f} seconds", flush = True)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        start = time.time()
        image_size = 0
        images = []
        for idx in self.dataset_indices:
            image = self.open_file_map[filename][f'records_{idx}'][sample_index]
            image_size += image.nbytes
            images.append(image)
        elapsed = time.time() - start

        # Sanity check that real data was read; outside the timer so the
        # min/max/mean over the sample don't count as read time.
        if self._args.my_rank == 0:
            for idx, image in zip(self.dataset_indices, images):
                print(f"{os.path.basename(filename)} records_{idx} idx={sample_index} "
                      f"shape={image.shape} min={image.min()} max={image.max()} "
                      f"mean={image.mean():.2f} nbytes={image.nbytes}", flush=True)

        print(f"[RANK {self._args.my_rank}] hdf5_reader: get_sample({os.path.basename(filename)}) total time: {elapsed:.4f} seconds", flush = True)
        dlp.update(image_size=image_size)
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
