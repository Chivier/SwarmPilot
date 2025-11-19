from numpy import allclose
import requests
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Literal, Optional, Dict
from pprint import pprint



class PlannerInput(BaseModel):
    """Input parameters for the optimization algorithm."""

    # Core parameters
    M: int = Field(..., description="Number of instances", gt=0)
    N: int = Field(..., description="Number of model types", gt=0)
    B: List[List[float]] = Field(..., description="Batch capacity matrix [M×N]")
    initial: Optional[List[int]] = Field(None, description="Initial deployment [M], -1 = no model")
    a: float = Field(..., description="Change constraint (0 < a ≤ 1)", gt=0, le=1)
    target: List[float] = Field(..., description="Target request distribution [N]")

    # Algorithm configuration
    algorithm: Literal["simulated_annealing", "integer_programming"] = Field(
        default="simulated_annealing",
        description="Optimization algorithm to use"
    )
    objective_method: Literal["relative_error", "ratio_difference", "weighted_squared"] = Field(
        default="relative_error",
        description="Objective function method"
    )
    verbose: bool = Field(default=True, description="Enable logging")

    # Simulated Annealing parameters
    initial_temp: float = Field(default=100.0, description="Starting temperature", gt=0)
    final_temp: float = Field(default=0.01, description="Ending temperature", gt=0)
    cooling_rate: float = Field(default=0.95, description="Temperature decay", gt=0, lt=1)
    max_iterations: int = Field(default=5000, description="Max iterations", gt=0)
    iterations_per_temp: int = Field(default=100, description="Iterations per temperature", gt=0)

    # Integer Programming parameters
    solver_name: str = Field(default="PULP_CBC_CMD", description="Solver backend")
    time_limit: int = Field(default=300, description="Timeout (seconds)", gt=0)

    @field_validator("B")
    @classmethod
    def validate_batch_capacity(cls, v, info):
        """Validate batch capacity matrix dimensions."""
        M = info.data.get("M")
        N = info.data.get("N")

        if M is not None and len(v) != M:
            raise ValueError(f"B must have {M} rows (M instances), got {len(v)}")

        if N is not None:
            for i, row in enumerate(v):
                if len(row) != N:
                    raise ValueError(f"B row {i} must have {N} columns (N models), got {len(row)}")
                if any(val < 0 for val in row):
                    raise ValueError(f"B row {i} contains negative values")

        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v, info):
        """Validate target distribution."""
        N = info.data.get("N")

        if N is not None and len(v) != N:
            raise ValueError(f"target must have length {N}, got {len(v)}")

        if any(val < 0 for val in v):
            raise ValueError("target contains negative values")

        return v

    @model_validator(mode="after")
    def validate_temperature_range(self):
        """Validate temperature parameters."""
        if self.final_temp >= self.initial_temp:
            raise ValueError("final_temp must be less than initial_temp")
        return self
      
class InstanceInfo(BaseModel):
    """Information about a target instance."""

    endpoint: str = Field(..., description="Instance API endpoint")
    current_model: str = Field(..., description="Current model name")

class DeploymentInput(BaseModel):
    """Input for deployment with optimization."""

    instances: List[InstanceInfo] = Field(..., description="Target instances")
    planner_input: PlannerInput = Field(..., description="Optimization config")
    scheduler_mapping: Dict[str, str] = Field(None, description="Scheduler URL for instance registration")

    @model_validator(mode="after")
    def validate_instances_match(self):
        """Validate that number of instances matches M."""
        if len(self.instances) != self.planner_input.M:
            raise ValueError(
                f"Number of instances ({len(self.instances)}) must match M ({self.planner_input.M})"
            )
        return self
      
def build_instance_info():
  instance_a_start_port = 8210
  instance_b_start_port = 8300
  
  instance_a_num = requests.get("http://localhost:8100/health").json()["stats"]["total_instances"]
  instance_b_num = requests.get("http://localhost:8200/health").json()["stats"]["total_instances"]
  
  print(f"Initial state: a: {instance_a_num}, b: {instance_b_num}")
  
  all_instances = []
  for i in range(instance_a_num):
    all_instances.append(
      InstanceInfo(
        endpoint=f"http://localhost:{instance_a_start_port + i}",
        current_model="sleep_model_a"
      )
    )
  
  b_instances = []
  for i in range(instance_b_num):
    all_instances.append(
      InstanceInfo(
        endpoint=f"http://localhost:{instance_b_start_port + i}",
        current_model="sleep_model_b"
      )
    )
    
  return all_instances, instance_a_num, instance_b_num
    
  
    
def build_planner_input(instance_a_num, instance_b_num) -> PlannerInput:
  return PlannerInput(
    M = instance_a_num + instance_b_num,
    N = 2,
    B = [[1, 30]] * (instance_a_num + instance_b_num),
    a = 1,
    target = [1, 10],
    algorithm = "simulated_annealing",
    objective_method = "ratio_difference"
  )  
  

def build_deployment_input():
  instance_info, num_a, num_b = build_instance_info()
  planner_input = build_planner_input(num_a, num_b)
  return DeploymentInput( 
    instances=instance_info,
    planner_input=planner_input,
    scheduler_mapping={
      "sleep_model_a": "http://localhost:8100",
      "sleep_model_b": "http://localhost:8200"
    }
  )
  
def perform_redeploy():
  planner_endpoint = "http://localhost:8202/deploy/migration"
  payload = build_deployment_input()
  pprint(payload.model_dump())
  response = requests.post(planner_endpoint, json=payload.model_dump())
  
  print(response.status_code)
 
perform_redeploy()