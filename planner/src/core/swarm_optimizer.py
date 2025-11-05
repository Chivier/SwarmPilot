"""
SwarmPilot 模型服务优化算法实现

该模块实现了两种优化算法来解决SwarmPilot模型部署优化问题：
1. 整数规划 (Integer Programming)
2. 模拟退火 (Simulated Annealing)

问题描述：
在有限的机器变更约束下，优化多台机器上的模型部署配置，
使得整体服务能力分布尽可能匹配实际请求分布。
"""

import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from loguru import logger

random.seed(42)
class SwarmOptimizer(ABC):
    """
    SwarmPilot优化器抽象基类

    定义了所有优化算法的通用接口和共享功能
    """

    def __init__(self, M: int, N: int, B: np.ndarray, initial: np.ndarray,
                 a: float, target: np.ndarray):
        """
        初始化优化器

        Args:
            M: 机器数量
            N: 模型数量
            B: 批处理能力矩阵 [M×N]，B[i][j]表示机器i对模型j的处理能力
            initial: 初始状态向量 [M]，initial[i]表示机器i初始部署的模型
            a: 变更因子，控制允许变更的机器数量上限
            target: 目标请求分布 [N]，期望的各模型请求比例
        """
        self.M = M  # 机器数量
        self.N = N  # 模型数量
        self.B = B.copy()  # 批处理能力矩阵
        self.initial = initial.copy()  # 初始部署状态
        self.a = a  # 变更因子
        self.target = target.copy()  # 目标请求分布
        self.max_changes = int(a * M)  # 最大允许变更数量

        # 验证输入参数
        self._validate_inputs()

        # 预处理：计算每个模型的可部署机器集合
        self.valid_assignments = self._precompute_valid_assignments()

    def _validate_inputs(self):
        """验证输入参数的有效性"""
        assert self.B.shape == (self.M, self.N), f"批处理能力矩阵维度错误: {self.B.shape} != ({self.M}, {self.N})"
        assert len(self.initial) == self.M, f"初始状态向量长度错误: {len(self.initial)} != {self.M}"
        assert len(self.target) == self.N, f"目标分布向量长度错误: {len(self.target)} != {self.N}"
        assert 0 < self.a <= 1, f"变更因子超出范围: {self.a} 不在 (0, 1]"

        # Support -1 as "no model deployed" initial state
        # -1 means the planner should compute the optimal initial deployment
        assert all(-1 <= x < self.N for x in self.initial), "初始状态包含无效模型编号"

        # Only validate capacity for VMs with deployed models (not -1)
        for i in range(self.M):
            if self.initial[i] != -1:
                assert self.B[i, self.initial[i]] > 0, \
                    f"初始状态包含无效部署: 机器 {i} 部署模型 {self.initial[i]} 但容量为 0"

    def _precompute_valid_assignments(self) -> Dict[int, List[int]]:
        """
        预计算每个模型的可部署机器集合

        Returns:
            Dict[model_id, List[machine_ids]]：每个模型可以部署的机器列表
        """
        valid_assignments = {}
        for j in range(self.N):
            valid_assignments[j] = [i for i in range(self.M) if self.B[i, j] > 0]
        return valid_assignments

    def generate_initial_deployment(self) -> np.ndarray:
        """
        为包含 -1 的初始状态生成一个有效的初始部署方案

        对于 current_model_id = -1 的 VM，根据 batch matrix 为其分配
        能力最高的模型。这样可以确保优化算法有一个有效的起点。

        Returns:
            有效的初始部署方案 [M]
        """
        deployment = self.initial.copy()

        # 为每个 -1 位置分配最优模型
        for i in range(self.M):
            if deployment[i] == -1:
                # 找到该 VM 上容量最高的模型
                best_model = -1
                best_capacity = 0
                for j in range(self.N):
                    if self.B[i, j] > best_capacity:
                        best_capacity = self.B[i, j]
                        best_model = j

                if best_model != -1:
                    deployment[i] = best_model
                else:
                    # 如果该 VM 对所有模型容量都为 0，分配第一个模型
                    # 这种情况应该在配置验证时被捕获
                    logger.warning(f"VM {i} 对所有模型容量都为 0，使用模型 0 作为默认值")
                    deployment[i] = 0

        logger.info(f"生成初始部署方案: {deployment}")
        return deployment

    def compute_service_capacity(self, deployment: np.ndarray) -> np.ndarray:
        """
        计算给定部署方案的服务能力分布

        Args:
            deployment: 部署方案 [M]，deployment[i]表示机器i部署的模型

        Returns:
            各模型的总服务能力 [N]
        """
        capacity = np.zeros(self.N)
        for i in range(self.M):
            model = deployment[i]
            # Skip VMs with no model deployed (model_id == -1)
            if model != -1:
                capacity[model] += self.B[i, model]
        return capacity

    def compute_changes(self, deployment: np.ndarray) -> int:
        """
        计算部署方案相对于初始状态的变更数量

        Args:
            deployment: 部署方案 [M]

        Returns:
            变更的机器数量
        """
        return np.sum(deployment != self.initial)

    def is_valid_deployment(self, deployment: np.ndarray) -> bool:
        """
        检查部署方案是否有效

        Args:
            deployment: 部署方案 [M]

        Returns:
            是否有效
        """
        # 检查是否超出变更限制
        if self.compute_changes(deployment) > self.max_changes:
            return False

        # 检查每台机器的部署是否可行
        for i in range(self.M):
            model = deployment[i]
            # -1 means no model deployed, which is invalid for final deployment
            # but valid during optimization process
            if model == -1:
                return False
            if self.B[i, model] == 0:
                return False

        return True

    def objective_function(self, deployment: np.ndarray, method: str = 'relative_error') -> float:
        """
        计算目标函数值（越小越好）

        Args:
            deployment: 部署方案 [M]
            method: 目标函数类型
                - 'relative_error': 相对误差最小化
                - 'ratio_difference': 比例差异最小化
                - 'weighted_squared': 加权平方误差

        Returns:
            目标函数值
        """
        if not self.is_valid_deployment(deployment):
            return float('inf')

        capacity = self.compute_service_capacity(deployment)
        target_sum = np.sum(self.target)
        capacity_sum = np.sum(capacity)

        if capacity_sum == 0:
            return float('inf')

        if method == 'relative_error':
            # 相对误差最小化
            capacity_ratio = capacity / capacity_sum
            target_ratio = self.target / target_sum
            return np.sum(np.abs(capacity_ratio - target_ratio))

        elif method == 'ratio_difference':
            # 比例差异最小化
            ratios = capacity / (self.target + 1e-8)  # 避免除零
            scale_factor = capacity_sum / target_sum
            return np.max(np.abs(ratios - scale_factor))

        elif method == 'weighted_squared':
            # 加权平方误差
            scale_factor = capacity_sum / target_sum
            scaled_target = scale_factor * self.target
            weights = 1.0 / (self.target + 1e-8)  # 权重与目标成反比
            return np.sum(weights * (capacity - scaled_target) ** 2)

        else:
            raise ValueError(f"未知的目标函数类型: {method}")

    @abstractmethod
    def optimize(self, **kwargs) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        执行优化算法

        Returns:
            (最优部署方案, 最优目标函数值, 统计信息)
        """
        pass




class SimulatedAnnealingOptimizer(SwarmOptimizer):
    """
    模拟退火优化器

    策略：允许接受较差解以跳出局部最优，温度逐渐降低
    具有更强的全局搜索能力
    """

    def optimize(self, objective_method: str = 'relative_error',
                 initial_temp: float = 100.0,
                 final_temp: float = 0.01,
                 cooling_rate: float = 0.95,
                 max_iterations: int = 5000,
                 iterations_per_temp: int = 100,
                 verbose: bool = True) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        执行模拟退火优化

        Args:
            objective_method: 目标函数类型
            initial_temp: 初始温度
            final_temp: 终止温度
            cooling_rate: 冷却率（每次温度乘以该因子）
            max_iterations: 最大迭代次数
            iterations_per_temp: 每个温度下的迭代次数
            verbose: 是否输出详细信息

        Returns:
            (最优部署方案, 最优目标函数值, 统计信息)
        """
        # Generate valid initial deployment if initial state contains -1
        if -1 in self.initial:
            logger.info("检测到初始状态包含 -1 (未部署), 自动生成初始部署方案")
            current_deployment = self.generate_initial_deployment()
            self.initial = current_deployment.copy()
        else:
            current_deployment = self.initial.copy()

        current_score = self.objective_function(current_deployment, objective_method)
        best_deployment = current_deployment.copy()
        best_score = current_score

        temperature = initial_temp
        iterations = 0
        acceptances = 0
        rejections = 0
        temperature_changes = 0

        if verbose:
            logger.info(f"模拟退火开始，初始温度: {initial_temp}，初始得分: {current_score:.6f}")

        while temperature > final_temp and iterations < max_iterations:
            temp_iterations = 0
            temp_acceptances = 0

            # 在当前温度下迭代
            while temp_iterations < iterations_per_temp and iterations < max_iterations:
                # 生成邻域解（随机单机器变更）
                neighbor = self._generate_random_neighbor(current_deployment)

                if neighbor is not None and self.is_valid_deployment(neighbor):
                    neighbor_score = self.objective_function(neighbor, objective_method)
                    delta = neighbor_score - current_score

                    # 接受准则：更好的解总是接受，较差的解按概率接受
                    if delta < 0 or random.random() < math.exp(-delta / temperature):
                        current_deployment = neighbor.copy()
                        current_score = neighbor_score
                        temp_acceptances += 1
                        acceptances += 1

                        # 更新全局最佳解
                        if neighbor_score < best_score:
                            best_deployment = neighbor.copy()
                            best_score = neighbor_score
                    else:
                        rejections += 1

                temp_iterations += 1
                iterations += 1

            # 降温
            temperature *= cooling_rate
            temperature_changes += 1

            if verbose and temperature_changes % 10 == 0:
                acceptance_rate = temp_acceptances / iterations_per_temp if iterations_per_temp > 0 else 0
                logger.info(f"温度: {temperature:.4f}，当前得分: {current_score:.6f}，"
                          f"最佳得分: {best_score:.6f}，接受率: {acceptance_rate:.3f}")

        stats = {
            'algorithm': 'simulated_annealing',
            'iterations': iterations,
            'temperature_changes': temperature_changes,
            'acceptances': acceptances,
            'rejections': rejections,
            'acceptance_rate': acceptances / (acceptances + rejections) if (acceptances + rejections) > 0 else 0,
            'final_temperature': temperature,
            'initial_score': self.objective_function(self.initial, objective_method),
            'final_score': best_score
        }

        if verbose:
            logger.info(f"模拟退火完成，总迭代: {iterations}，接受率: {stats['acceptance_rate']:.3f}，"
                       f"最终得分: {best_score:.6f}")

        return best_deployment, best_score, stats

    def _generate_random_neighbor(self, deployment: np.ndarray) -> Optional[np.ndarray]:
        """
        生成随机邻域解

        Args:
            deployment: 当前部署方案

        Returns:
            随机邻域解，如果无法生成则返回None
        """
        # 随机选择一台机器
        machine = random.randint(0, self.M - 1)
        current_model = deployment[machine]

        # 获取该机器可部署的其他模型
        valid_models = [m for m in range(self.N)
                       if m != current_model and self.B[machine, m] > 0]

        if not valid_models:
            return None

        # 随机选择新模型
        new_model = random.choice(valid_models)

        neighbor = deployment.copy()
        neighbor[machine] = new_model

        return neighbor


try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    pulp = None


class IntegerProgrammingOptimizer(SwarmOptimizer):
    """
    整数规划优化器

    使用线性规划求解器求解混合整数规划问题
    需要安装 pulp 库：pip install pulp
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not PULP_AVAILABLE:
            raise ImportError("需要安装 pulp 库来使用整数规划优化器：pip install pulp")

    def optimize(self, objective_method: str = 'relative_error',
                 solver_name: str = 'PULP_CBC_CMD',
                 time_limit: int = 300,
                 verbose: bool = True) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        执行整数规划优化

        Args:
            objective_method: 目标函数类型（注意：整数规划只支持线性目标函数）
            solver_name: 求解器名称
            time_limit: 时间限制（秒）
            verbose: 是否输出详细信息

        Returns:
            (最优部署方案, 最优目标函数值, 统计信息)
        """
        if verbose:
            logger.info(f"开始整数规划优化，求解器: {solver_name}")

        # 创建问题实例
        prob = pulp.LpProblem("SwarmPilot_Optimization", pulp.LpMinimize)

        # 决策变量：x[i][j] 表示机器i是否部署模型j
        x = {}
        for i in range(self.M):
            for j in range(self.N):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')

        # 变更指示变量：y[i] 表示机器i是否发生变更
        y = {}
        for i in range(self.M):
            y[i] = pulp.LpVariable(f"y_{i}", cat='Binary')

        # 约束1：每台机器只能部署一个模型
        for i in range(self.M):
            prob += pulp.lpSum([x[i, j] for j in range(self.N)]) == 1

        # 约束2：机器能力约束
        for i in range(self.M):
            for j in range(self.N):
                if self.B[i, j] == 0:
                    prob += x[i, j] == 0

        # 约束3：变更检测
        # If initial_model is -1 (no model deployed), treat it as a change
        for i in range(self.M):
            initial_model = self.initial[i]
            if initial_model == -1:
                # No initial model, any deployment counts as a change
                # y[i] = 1 means changed (must deploy a model)
                prob += y[i] == 1
            else:
                # y[i] >= 1 - x[i, initial_model] means:
                # if x[i, initial_model] = 0 (changed), then y[i] >= 1 (must be 1)
                prob += y[i] >= 1 - x[i, initial_model]

        # 约束4：变更数量限制
        prob += pulp.lpSum([y[i] for i in range(self.M)]) <= self.max_changes

        # 目标函数：这里使用简化的线性目标函数
        # 由于整数规划难以处理复杂的非线性目标函数，我们使用线性近似
        if objective_method == 'relative_error':
            # 最小化各模型服务能力与目标的加权偏差
            target_sum = np.sum(self.target)
            target_ratio = self.target / target_sum

            # 计算各模型的服务能力
            capacity = {}
            for j in range(self.N):
                capacity[j] = pulp.lpSum([self.B[i, j] * x[i, j] for i in range(self.M)])

            total_capacity = pulp.lpSum([capacity[j] for j in range(self.N)])

            # 使用线性近似的目标函数
            # 最小化 |capacity[j]/total_capacity - target_ratio[j]|
            # 由于绝对值不是线性的，我们使用惩罚项
            deviation_vars = {}
            for j in range(self.N):
                deviation_vars[j] = pulp.LpVariable(f"dev_{j}", lowBound=0)
                # 这是一个简化，实际实现可能需要更复杂的线性化技术
                prob += deviation_vars[j] >= capacity[j] - target_ratio[j] * total_capacity
                prob += deviation_vars[j] >= target_ratio[j] * total_capacity - capacity[j]

            prob += pulp.lpSum([deviation_vars[j] for j in range(self.N)])

        else:
            # 对于其他目标函数，使用基本的负载均衡目标
            logger.warning(f"整数规划不支持目标函数 {objective_method}，使用默认线性目标函数")
            capacity = {}
            for j in range(self.N):
                capacity[j] = pulp.lpSum([self.B[i, j] * x[i, j] for i in range(self.M)])

            # 最小化变更数量（作为简单的目标函数）
            prob += pulp.lpSum([y[i] for i in range(self.M)])

        # 求解
        try:
            if solver_name == 'PULP_CBC_CMD':
                solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
            else:
                solver = pulp.getSolver(solver_name, timeLimit=time_limit, msg=verbose)

            prob.solve(solver)

            # 检查求解状态
            status = pulp.LpStatus[prob.status]

            if status == 'Optimal':
                # 提取解
                deployment = np.zeros(self.M, dtype=int)
                for i in range(self.M):
                    for j in range(self.N):
                        if x[i, j].varValue is not None and x[i, j].varValue > 0.5:
                            deployment[i] = j
                            break

                final_score = self.objective_function(deployment, objective_method)

                stats = {
                    'algorithm': 'integer_programming',
                    'solver': solver_name,
                    'status': status,
                    'objective_value': pulp.value(prob.objective),
                    'solve_time': prob.solutionTime if hasattr(prob, 'solutionTime') else None,
                    'initial_score': self.objective_function(self.initial, objective_method),
                    'final_score': final_score
                }

                if verbose:
                    logger.info(f"整数规划求解成功，状态: {status}，最终得分: {final_score:.6f}")

                return deployment, final_score, stats

            else:
                if verbose:
                    logger.warning(f"整数规划求解失败，状态: {status}")

                # 返回初始解
                initial_score = self.objective_function(self.initial, objective_method)
                stats = {
                    'algorithm': 'integer_programming',
                    'solver': solver_name,
                    'status': status,
                    'objective_value': None,
                    'solve_time': None,
                    'initial_score': initial_score,
                    'final_score': initial_score
                }

                return self.initial.copy(), initial_score, stats

        except Exception as e:
            logger.error(f"整数规划求解过程中出现错误: {str(e)}")
            initial_score = self.objective_function(self.initial, objective_method)
            stats = {
                'algorithm': 'integer_programming',
                'solver': solver_name,
                'status': 'Error',
                'error': str(e),
                'initial_score': initial_score,
                'final_score': initial_score
            }

            return self.initial.copy(), initial_score, stats


def compare_algorithms(M: int, N: int, B: np.ndarray, initial: np.ndarray,
                      a: float, target: np.ndarray,
                      objective_method: str = 'relative_error',
                      verbose: bool = True) -> Dict[str, Any]:
    """
    比较两种优化算法的性能

    Args:
        M, N, B, initial, a, target: 问题参数
        objective_method: 目标函数类型
        verbose: 是否输出详细信息

    Returns:
        包含算法结果的字典
    """
    results = {}

    # 测试模拟退火算法
    try:
        if verbose:
            print("\n" + "="*50)
            print("测试模拟退火算法")
            print("="*50)

        sa_opt = SimulatedAnnealingOptimizer(M, N, B, initial, a, target)
        deployment, score, stats = sa_opt.optimize(objective_method, verbose=verbose)
        results['simulated_annealing'] = {
            'deployment': deployment,
            'score': score,
            'stats': stats
        }
    except Exception as e:
        logger.error(f"模拟退火算法执行失败: {str(e)}")
        results['simulated_annealing'] = {'error': str(e)}

    # 测试整数规划算法（如果可用）
    if PULP_AVAILABLE:
        try:
            if verbose:
                print("\n" + "="*50)
                print("测试整数规划算法")
                print("="*50)

            ip_opt = IntegerProgrammingOptimizer(M, N, B, initial, a, target)
            deployment, score, stats = ip_opt.optimize(objective_method, verbose=verbose)
            results['integer_programming'] = {
                'deployment': deployment,
                'score': score,
                'stats': stats
            }
        except Exception as e:
            logger.error(f"整数规划算法执行失败: {str(e)}")
            results['integer_programming'] = {'error': str(e)}
    else:
        results['integer_programming'] = {'error': 'pulp库未安装'}

    return results


if __name__ == "__main__":
    # 运行示例
    print("SwarmPilot 模型服务优化算法演示")
    print("="*60)

    # 示例问题参数
    M = 4  # 4台机器
    N = 3  # 3个模型
    B = np.array([
        [10, 5, 0],   # 机器0
        [8, 6, 4],    # 机器1
        [0, 10, 8],   # 机器2
        [6, 0, 12]    # 机器3
    ])
    initial = np.array([0, 1, 2, 2])  # 初始部署
    a = 0.5  # 最多改变50%的机器
    target = np.array([20, 30, 25])  # 期望比例 20:30:25

    print(f"问题规模: {M}台机器, {N}个模型")
    print(f"初始部署: {initial}")
    print(f"目标分布: {target}")
    print(f"最大变更数: {int(a * M)}")

    # 比较两种算法性能
    results = compare_algorithms(M, N, B, initial, a, target, verbose=True)

    # 输出比较结果
    print("\n" + "="*60)
    print("算法性能比较")
    print("="*60)

    for alg_name, result in results.items():
        if 'error' in result:
            print(f"{alg_name}: 执行失败 - {result['error']}")
        else:
            print(f"{alg_name}:")
            print(f"  最终部署: {result['deployment']}")
            print(f"  目标函数值: {result['score']:.6f}")
            print(f"  改进程度: {result['stats']['initial_score'] - result['score']:.6f}")
            print()