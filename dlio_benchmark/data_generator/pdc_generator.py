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

import logging

import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress

from pdc_ctypes import get_client

dlp = Profile(MODULE_DATA_GENERATOR)

# Must match pdc_reader.py
RECORD_DTYPE = np.uint8

import os

"""
PDC data generator for DLIO, via ctypes (no PDCpy). 
Selected with dataset.format: pdc
"""

def object_name(path):
    return os.path.basename(path)

class PDCGenerator(DataGenerator):
    # Generates one PDC object per DLIO file

    def __init__(self):
        super().__init__()

    @dlp.log
    def generate(self):
        super().generate()

        np.random.seed(10)
        record_labels = np.zeros(self.num_samples, dtype=np.int32)
        dim = self.get_dimension(self.total_files_to_generate)

        client = get_client("dlio")

        # print(f"[gen] record_length={self._args.record_length!r} "
        #       f"num_samples={self.num_samples} "
        #       f"total_files={self.total_files_to_generate} "
        #       f"dim[0:4]={dim[:4]} len(dim)={len(dim)}", flush=True)
        # print("[gen] args:", [f"{k}={v}" for k, v in vars(self._args).items()
        #                       if "record" in k or "dim" in k], flush=True)

        for i in dlp.iter(range(self.my_rank,
                                int(self.total_files_to_generate),
                                self.comm_size)):
            dim1 = int(dim[2 * i])
            dim2 = int(dim[2 * i + 1])

            # samples-first: sample k is a contiguous region
            records = np.random.randint(
                255, size=(self.num_samples, dim1, dim2), dtype=RECORD_DTYPE
            )
            out_path_spec = self.storage.get_uri(self._file_list[i])

            
            progress(i + 1, self.total_files_to_generate,
                     f"Generating PDC Data - data size {records.nbytes}")

            name = object_name(out_path_spec)
            cont_id = client.create_container(name)
            client.create_object(cont_id, name, records)
            client.create_object(cont_id, f"{name}-labels", record_labels)
            client.create_object(cont_id, f"{name}-dims", np.array([self.num_samples, dim1, dim2], dtype=np.int64))

            print(f"file genterated: {name}")

            # cont_id = client.create_container(out_path_spec)
            # client.create_object(cont_id, out_path_spec, records)
            # client.create_object(cont_id, f"{out_path_spec}-labels", record_labels)
            logging.info(f"PDC object created {name}")

            # Placeholder file so DLIO's file list still works
            with open(out_path_spec, "w"):
                pass

        np.random.seed()