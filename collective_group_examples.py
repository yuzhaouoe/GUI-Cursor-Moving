"""
Examples of how to add the main process to Ray collective groups.

Ray collective groups only work with Ray actors, not the driver process (main).
Here are different approaches to handle this:
"""

import ray
from ray.experimental.collective import create_collective_group
import torch


# ============================================================================
# OPTION 1: Coordinator Actor (Recommended)
# ============================================================================
@ray.remote
class SenderActor:
    def __init__(self):
        pass
    
    @ray.method(tensor_transport="NIXL")
    def send_data(self):
        return torch.rand(100, 100)


@ray.remote
class ReceiverActor:
    def __init__(self):
        pass
    
    def receive_data(self, data):
        return torch.sum(data)


@ray.remote
class CoordinatorActor:
    """Acts as the main process within the collective group"""
    def __init__(self):
        pass
    
    def orchestrate(self, sender, receiver):
        """This method can participate in collective operations"""
        data = sender.send_data.remote()
        result = receiver.receive_data.remote(data)
        return ray.get(result)


def example_with_coordinator():
    """Main process creates actors and lets coordinator handle the workflow"""
    sender = SenderActor.remote()
    receiver = ReceiverActor.remote()
    coordinator = CoordinatorActor.remote()
    
    # Create collective group with all actors including coordinator
    group = create_collective_group([sender, receiver, coordinator], backend="NIXL")
    
    # Run workflow through coordinator
    result = ray.get(coordinator.orchestrate.remote(sender, receiver))
    print(f"Result: {result}")


# ============================================================================
# OPTION 2: Main process stays outside (Current pattern - No collective ops)
# ============================================================================
def example_without_main_in_group():
    """Main process orchestrates but doesn't participate in collective ops"""
    sender = SenderActor.remote()
    receiver = ReceiverActor.remote()
    
    # Only actors in the collective group
    group = create_collective_group([sender, receiver], backend="NIXL")
    
    # Main process orchestrates from outside
    data = sender.send_data.remote()
    result = receiver.receive_data.remote(data)
    print(f"Result: {ray.get(result)}")


# ============================================================================
# OPTION 3: Using ray.util.collective for more control
# ============================================================================
@ray.remote
class CollectiveWorker:
    def __init__(self, rank):
        self.rank = rank
    
    def init_collective(self, world_size, group_name="default"):
        """Initialize collective group from within the actor"""
        from ray.util.collective import collective
        # This would be called on all workers
        pass
    
    def collective_broadcast(self, tensor, src_rank, group_name="default"):
        """Perform broadcast operation"""
        from ray.util.collective import collective
        collective.broadcast(tensor, src_rank=src_rank, group_name=group_name)
        return tensor


def example_with_util_collective():
    """Using ray.util.collective for more granular control"""
    from ray.util.collective import collective
    
    workers = [CollectiveWorker.remote(rank=i) for i in range(3)]
    
    # Create collective group using ray.util.collective
    collective.create_collective_group(
        workers,
        world_size=3,
        ranks=list(range(3)),
        backend="nccl",
        group_name="my_group"
    )
    
    # Main process can coordinate, but collective ops happen in actors
    # workers would call collective operations internally


# ============================================================================
# SUMMARY
# ============================================================================
"""
Key Points:

1. **Cannot add driver process**: Ray's collective groups only work with actors,
   not the main/driver process.

2. **Best approach**: Create a CoordinatorActor that acts as your "main process"
   within the collective group. The actual main process just sets things up
   and waits for results.

3. **Why?**: Collective operations require all participants to be on equal
   footing with GPU/tensor operations. The driver process is fundamentally
   different from worker processes.

4. **Alternative**: If you need collective operations, use ray.util.collective
   and have all logic inside actors. The main process only orchestrates.

5. **NIXL backend**: Optimized for tensor transfer between actors, but still
   requires all participants to be actors.
"""


if __name__ == "__main__":
    ray.init()
    
    print("Example 1: With Coordinator Actor")
    example_with_coordinator()
    
    print("\nExample 2: Main process outside group")
    example_without_main_in_group()
    
    ray.shutdown()
