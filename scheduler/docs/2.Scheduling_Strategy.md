# Supported Scheduling Strategy

## Strategies

### Round Robin

#### Queue status management

Current ID: current selected instance id
Next ID: next selected instance id

#### Instance Selection

Send task to each task with a round-robin order.

### Shortest Queue

#### Queue status management

For each queue:
expect: float
error: float

#### Instance Selection

Select the queue with lowest expect, for the same expect, select the queue with minmium error.

#### Queue Update

When schedule a task to specific queue:
update the expect and error value with this task's expect and error based on the Error Propagation Formula

Expect：
$$ E(Y) = E(X_1) + E(X_2) = \mu_1 + \mu_2 $$
Var：
$$ \text{Var}(Y) = \text{Var}(X_1) + \text{Var}(X_2) = \sigma_1^2 + \sigma_2^2 $$

When a task is finished:
update the expect of specific instance with the real execution time of the task, but don't alter the error

### Probabilistic

Select the queue based on the distribution of queue's runtime

#### Queue status management

For each queue, use quantile distribution:
quantiles: List[float]
values: List[float]

#### Instance Selection

for each queue, do a random sample on it, then select the queue with lowest sample value

#### Queue update

When schedule a task to specific queue:
update the value of quantile with the value of quantile of the task with Monte Carlo Method

When a task is finished:
update the value of quantile with the value of quantile of the task with Monte Carlo Method

Important: value of quantile of specific task should be cached in the scheduler 