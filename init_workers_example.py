from ast import main
import os

os.environ["RAY_rdt_fetch_fail_timeout_milliseconds"] = f"{30 * 60 * 1000}"
import torch
import ray
from ray.experimental.collective import create_collective_group
import time
from PIL import Image
import sys
from verl import DataProto
from cursor.rl.dist_dataproto import DistDataProto
import numpy as np

@ray.remote
class SenderActor:
    def __init__(self, tensor_shape=None, my_name=None):
        self.tensor_shape: tuple = tensor_shape
        self.my_name: str = my_name

    @ray.method(tensor_transport="NIXL")
    def fetch_dataproto(self, idx=0):
        create_tensor_start_time = time.perf_counter()
        rand_tensor = torch.rand(self.tensor_shape)
        rand_tensor += idx  # make different tensors
        create_tensor_end_time = time.perf_counter()
        gbn = (rand_tensor.element_size() * rand_tensor.numel()) / (1024 ** 3)
        data = DataProto.from_single_dict({"tensor": rand_tensor})
        print(f"{self.my_name}: Time taken to create tensor ({gbn:.2f} GB): {create_tensor_end_time - create_tensor_start_time:.2f}s")
        return data # "tensor": rand_tensor, "image": None}

    @ray.method(tensor_transport="NIXL")
    def fetch_dist_dataproto(self, idx=0):
        create_tensor_start_time = time.perf_counter()
        rand_tensor = torch.rand(self.tensor_shape)
        rand_tensor += idx  # make different tensors
        create_tensor_end_time = time.perf_counter()
        gbn = (rand_tensor.element_size() * rand_tensor.numel()) / (1024 ** 3)
        data = DistDataProto.from_single_dict({"tensor": rand_tensor})
        data.non_tensor_batch["image"] = [torch.rand(self.tensor_shape) for _ in range(1)]
        print(f"{self.my_name}: Time taken to create tensor ({gbn:.2f} GB): {create_tensor_end_time - create_tensor_start_time:.2f}s")
        return data # "tensor": rand_tensor, "image": None}
    
    @ray.method(tensor_transport="NIXL")
    def fetch_tensor(self, idx=0):
        create_tensor_start_time = time.perf_counter()
        rand_tensor = torch.rand(self.tensor_shape)
        rand_tensor += idx  # make different tensors
        create_tensor_end_time = time.perf_counter()
        gbn = (rand_tensor.element_size() * rand_tensor.numel()) / (1024 ** 3)
        print(f"{self.my_name}: Time taken to create tensor ({gbn:.2f} GB): {create_tensor_end_time - create_tensor_start_time:.2f}s")
        return {"tensor": rand_tensor, "image": None}


@ray.remote
class OperateActor:
    def __init__(self, operator=None, my_name=None):
        self.operator: str = operator
        self.my_name: str = my_name

    def operate(self, item):

        if type(item) is not dict:
            assert type(item) in [DataProto, DistDataProto]
            tensor = item.batch["tensor"]
        else:
            tensor = item["tensor"]
        start_op_time = time.perf_counter()
        if self.operator == "sum":
            res = torch.sum(tensor)
        elif self.operator == "mean":
            res = torch.mean(tensor)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
        res = res + 1
        end_op_time = time.perf_counter()
        print(f"{self.my_name}: Time taken for {self.operator} op: {end_op_time - start_op_time:.2f}s")
        return res


@ray.remote
class CoordinatorActor:
    """Coordinator actor that acts as the main process in the collective group"""
    def __init__(self, my_name=None):
        self.my_name: str = my_name

    def run_workflow(self, sender, receiver):
        """Execute the main workflow logic"""
        start_fetch_time = time.perf_counter()
        
        # list_item_ref = [sender.fetch_tensor.remote(idx=idx) for idx in range(3)]
        # list_item = [sender.fetch_dataproto.remote(idx=idx) for idx in range(3)]
        list_item_ref = [sender.fetch_dist_dataproto.remote(idx=idx) for idx in range(3)]

        # list_item = ray.get(list_item_ref, _tensor_transport="NIXL")
        results = [receiver.operate.remote(item) for item in list_item_ref]
        
        final_results = ray.get(results)
        end_op_time = time.perf_counter()
        
        print("Final result:", final_results)
        print(f"Program time: {end_op_time - start_fetch_time:.2f}s")
        return final_results


def main():

    sender = SenderActor.remote(tensor_shape=(10000, 100000), my_name="TestSender")
    receiver = OperateActor.remote(operator="sum", my_name="GreatReceiver")
    coordinator = CoordinatorActor.remote(my_name="MainCoordinator")

    # Now you can add the coordinator to the collective group
    group = create_collective_group([sender, receiver, coordinator], backend="NIXL")

    # Run the workflow through the coordinator
    results = ray.get(coordinator.run_workflow.remote(sender, receiver))


if __name__ == "__main__":
    main()