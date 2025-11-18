from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import random



class AvailableInstance(BaseModel):
  model_id: str
  endpoint: str
  
  
class AvailableInstanceStore:
  """
  Store for available instances.
  This class is designed for migration-based re-deployment.
  """
  def __init__(self):
    self.available_instances: Dict[str, List[AvailableInstance]] = {}
    self.lock = asyncio.Lock()

  async def get_available_instances_by_model_id(self, model_id: str) -> List[AvailableInstance]:
    async with self.lock:
      return self.available_instances.get(model_id, [])
    
  async def get_available_instances(self) -> List[AvailableInstance]:
    async with self.lock:
        return [instance for instances in self.available_instances.values() for instance in instances]
    
  async def add_available_instance(self, instance: AvailableInstance):
    async with self.lock:
      self.available_instances[instance.model_id] = self.available_instances.get(instance.model_id, []) + [instance]
    
  async def remove_available_instance(self, instance: AvailableInstance):
    async with self.lock:
      if instance.model_id in self.available_instances:
        self.available_instances[instance.model_id] = [
          i for i in self.available_instances[instance.model_id]
          if i.endpoint != instance.endpoint
        ]

  async def fetch_one_available_instance(self, model_id: str) -> Optional[AvailableInstance]:
    async with self.lock:
      if model_id not in self.available_instances or not self.available_instances[model_id]:
        return None
      selected_instance = self.available_instances[model_id][0]
      self.available_instances[model_id] = self.available_instances[model_id][1:]
      return selected_instance


_available_model_store: Optional[AvailableInstanceStore] = None

def get_available_instance_store() -> AvailableInstanceStore:
  global _available_model_store
  if _available_model_store is None:
    _available_model_store = AvailableInstanceStore()
  return _available_model_store