from verl import DataProto

class DistDataProto(DataProto):
    """A DataProto subclass for distributed data."""

    def __getstate__(self):
        return self.batch, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        batch, non_tensor_batch, meta_info = data

        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info