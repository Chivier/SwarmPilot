from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import random
from loguru import logger
from queue import Queue



class AvailableInstance(BaseModel):
  model_id: str
  endpoint: str
  
  
class AvailableInstanceStore:
  """
  Store for available instances.
  This class is designed for migration-based re-deployment.
  """
  def __init__(self):
    self.available_instances: Dict[str, Queue[AvailableInstance]] = {}
    self.lock = asyncio.Lock()

  async def get_available_instances_by_model_id(self, model_id: str) -> List[AvailableInstance]:
    async with self.lock:
      return self.available_instances.get(model_id, Queue())
    
  async def get_available_instances(self) -> List[AvailableInstance]:
    async with self.lock:
        return [instance for queue in self.available_instances.values() for instance in queue]
    
  async def add_available_instance(self, instance: AvailableInstance):
    async with self.lock:
      if not self.available_instances.get(instance.model_id):
        self.available_instances[instance.model_id] = Queue()
      self.available_instances[instance.model_id].put(instance)
    
  async def remove_available_instance(self, instance: AvailableInstance):
    async with self.lock:
      if instance.model_id in self.available_instances:
        self.available_instances[instance.model_id].remove(instance)

  async def fetch_one_available_instance(self, model_id: str) -> Optional[AvailableInstance]:
    async with self.lock:
      if model_id not in self.available_instances:
        logger.error(f"Model {model_id} not found in available instances")
        logger.error(f"Available instances: {self.available_instances.keys()}")
        return None
      if self.available_instances[model_id].empty():
        logger.error(f"Model {model_id} has no available instances, queue is empty")
        return None
      return self.available_instances[model_id].get()


_available_model_store: Optional[AvailableInstanceStore] = None

def get_available_instance_store() -> AvailableInstanceStore:
  global _available_model_store
  if _available_model_store is None:
    _available_model_store = AvailableInstanceStore()
  return _available_model_store