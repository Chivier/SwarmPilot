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
    scheduler_mapping: Dict[str, str] = Field(None, description="Model to scheduler URL mapping (default)")
    instance_scheduler_mapping: Dict[str, str] = Field(None, description="Per-instance scheduler URL mapping")

    @model_validator(mode="after")
    def validate_instances_match(self):
        """Validate that number of instances matches M."""
        if len(self.instances) != self.planner_input.M:
            raise ValueError(
                f"Number of instances ({len(self.instances)}) must match M ({self.planner_input.M})"
            )
        return self
      
def build_instance_info():
  # Service endpoints
  SCHEDULER_A_HOST = "29.209.114.51"
  SCHEDULER_B_HOST = "29.209.113.228"
  PLANNER_HOST = "20.209.114.166"
  PREDICTOR_HOST = "29.209.113.113"
  SERVICE_PORT = 8100

  # Instance hosts - Group A (llm_service_large_model)
  SLEEP_MODEL_A_HOSTS = [
    "29.209.106.237",
    "29.209.114.56",
    "29.209.114.241",
    "29.209.112.177",
    "29.209.113.235",
    "29.209.105.60",
  ]

  # Instance hosts - Group B (llm_service_small_model)
  SLEEP_MODEL_B_HOSTS = [
    "29.209.113.166",
    "29.209.113.176",
    "29.209.113.169",
    "29.209.112.74",
    "29.209.115.174",
    "29.209.113.156",
  ]

  # Port lists
  SCHEDULER_PORT_LIST = [8200, 8201, 8202, 8203]  # Register to scheduler
  PLANNER_PORT_LIST = [8204, 8205, 8206, 8207]    # Register to planner
  ALL_PORTS = SCHEDULER_PORT_LIST + PLANNER_PORT_LIST

  all_instances = []
  instance_scheduler_map = {}  # Maps instance endpoint to its scheduler URL

  # Build Group A instances (llm_service_large_model)
  for host in SLEEP_MODEL_A_HOSTS:
    # First half: register to scheduler
    for port in SCHEDULER_PORT_LIST:
      endpoint = f"http://{host}:{port}"
      all_instances.append(
        InstanceInfo(
          endpoint=endpoint,
          current_model="llm_service_large_model"
        )
      )
      instance_scheduler_map[endpoint] = f"http://{SCHEDULER_A_HOST}:{SERVICE_PORT}"

    # # Second half: register to planner
    # for port in PLANNER_PORT_LIST:
    #   endpoint = f"http://{host}:{port}"
    #   all_instances.append(
    #     InstanceInfo(
    #       endpoint=endpoint,
    #       current_model="llm_service_large_model"
    #     )
    #   )
    #   instance_scheduler_map[endpoint] = f"http://{PLANNER_HOST}:{SERVICE_PORT}"

  # Build Group B instances (llm_service_small_model)
  for host in SLEEP_MODEL_B_HOSTS:
    # First half: register to scheduler
    for port in SCHEDULER_PORT_LIST:
      endpoint = f"http://{host}:{port}"
      all_instances.append(
        InstanceInfo(
          endpoint=endpoint,
          current_model="llm_service_small_model"
        )
      )
      instance_scheduler_map[endpoint] = f"http://{SCHEDULER_B_HOST}:{SERVICE_PORT}"

    # # Second half: register to planner
    # for port in PLANNER_PORT_LIST:
    #   endpoint = f"http://{host}:{port}"
    #   all_instances.append(
    #     InstanceInfo(
    #       endpoint=endpoint,
    #       current_model="llm_service_small_model"
    #     )
    #   )
    #   instance_scheduler_map[endpoint] = f"http://{PLANNER_HOST}:{SERVICE_PORT}"

  instance_a_num = len(SLEEP_MODEL_A_HOSTS) * len(ALL_PORTS)
  instance_b_num = len(SLEEP_MODEL_B_HOSTS) * len(ALL_PORTS)

  print(f"Initial state: Group A (large_model): {instance_a_num}, Group B (small_model): {instance_b_num}")
  print(f"Total instances: {len(all_instances)}")
  print(f"  - Registered to scheduler: {len(SCHEDULER_PORT_LIST) * (len(SLEEP_MODEL_A_HOSTS) + len(SLEEP_MODEL_B_HOSTS))}")
  print(f"  - Registered to planner: {len(PLANNER_PORT_LIST) * (len(SLEEP_MODEL_A_HOSTS) + len(SLEEP_MODEL_B_HOSTS))}")

  return all_instances, instance_a_num, instance_b_num, instance_scheduler_map
    
  
    
def build_planner_input(instance_a_num, instance_b_num) -> PlannerInput:
  return PlannerInput(
    M = instance_a_num + instance_b_num,
    N = 2,
    B = [[1, 15]] * (instance_a_num + instance_b_num),
    a = 1,
    target = [1, 10],
    algorithm = "simulated_annealing",
    objective_method = "ratio_difference"
  )  
  

def build_deployment_input():
  instance_info, num_a, num_b, instance_scheduler_map = build_instance_info()
  planner_input = build_planner_input(num_a, num_b)
  return DeploymentInput(
    instances=instance_info,
    planner_input=planner_input,
    scheduler_mapping={
      "llm_service_large_model": "http://29.209.114.51:8100",
      "llm_service_small_model": "http://29.209.113.228:8100"
    },
    instance_scheduler_mapping=instance_scheduler_map
  )
  
def perform_redeploy():
  planner_endpoint = "http://20.209.114.166:8100/deploy/migration"
  payload = build_deployment_input()
  pprint(payload.model_dump())
  response = requests.post(planner_endpoint, json=payload.model_dump())
  
  print(response.status_code)
 
perform_redeploy()