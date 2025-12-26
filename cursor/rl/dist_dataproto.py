from verl import DataProto
import numpy as np

class DistDataProto(DataProto):
    """A DataProto subclass for distributed data."""

    def __getstate__(self):
        return self.batch, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        batch, non_tensor_batch, meta_info = data

        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info
    
    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                if key != "multi_modal_inputs":
                    assert isinstance(val, np.ndarray)

        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                if key != "multi_modal_inputs":
                    assert isinstance(val, np.ndarray), (
                        f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for "
                        f"{key=}, got {type(val)=}"
                    )
                    assert val.shape[0] == batch_size, (
                        f"key {key} length {len(val)} is not equal to batch size {batch_size}"
                    )
                else:
                    assert len(val) == batch_size