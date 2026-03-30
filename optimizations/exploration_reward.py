"""
优化方案4：自适应探索奖励系统
原问题：探索奖励机制过于粗糙，简单的坐标访问二值奖励

改进方案：
1. 基于区域难度的动态奖励
2. 结合访问频率和地点重要性
3. 使用好奇心驱动的探索
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Set
import heapq

class AdaptiveExplorationReward:
    """
    自适应探索奖励计算器
    根据位置难度和访问历史动态计算探索奖励
    """
    
    def __init__(self, grid_width: int = 20, grid_height: int = 18):
        """
        Args:
            grid_width: 地图宽度（网格数）
            grid_height: 地图高度（网格数）
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # 访问计数
        self.visit_counts = defaultdict(int)
        
        # 区域难度地图（预定义的难度值）
        self.difficulty_map = self._create_difficulty_map()
        
        # 访问历史用于计算新颖性
        self.visit_history = []
        self.max_history = 100
        
        # 探索边界（未知区域）
        self.frontier_tiles = set()
        
        # 统计信息
        self.total_visits = 0
        self.unique_tiles_visited = 0
    
    def _create_difficulty_map(self) -> Dict[Tuple[int, int], float]:
    
        difficulty_map = {}
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # 起始位置难度低
                distance_from_start = np.sqrt((x - 2)**2 + (y - 1)**2)
                
                # 基础难度（基于距离）
                base_difficulty = np.clip(distance_from_start / 15, 0, 1)
                
                # 添加一些特殊区域的难度调整
                # 森林区（迷宫）难度更高
                if 4 <= x <= 8 and 1 <= y <= 4:
                    base_difficulty *= 1.5
                
                # 洞穴区难度更高
                if 2 <= x <= 5 and 10 <= y <= 13:
                    base_difficulty *= 1.8
                
                difficulty_map[(x, y)] = np.clip(base_difficulty, 0, 1)
        
        return difficulty_map
    
    def get_position_novelty(self, pos: Tuple[int, int]) -> float:
      
        visit_count = self.visit_counts[pos]
        
        # 基于访问计数的新颖性衰减
        # 第一次访问：1.0，之后快速衰减
        novelty = 1.0 / (1.0 + np.log1p(visit_count))
        
        return novelty
    
    def get_exploration_reward(self, pos: Tuple[int, int], current_map: int) -> float:
     
        # 获取基本属性
        novelty = self.get_position_novelty(pos)
        difficulty = self.difficulty_map.get(pos, 0.5)
        
        # 如果是新位置，给予基础奖励
        if self.visit_counts[pos] == 0:
            exploration_reward = 0.05 * (1 + difficulty)
            self.unique_tiles_visited += 1
        else:
            # 已访问位置给予递减奖励
            exploration_reward = 0.005 * novelty * difficulty
        
        # 更新访问计数
        self.visit_counts[pos] += 1
        self.total_visits += 1
        
        # 记录访问历史
        self.visit_history.append(pos)
        if len(self.visit_history) > self.max_history:
            self.visit_history.pop(0)
        
        return max(exploration_reward, 0)
    
    def get_frontier_distance_reward(self, pos: Tuple[int, int]) -> float:
     
        # 找出所有未访问的相邻位置
        frontier = self._find_frontier_tiles()
        
        if not frontier:
            return 0
        
        # 计算到最近边界的距离
        min_distance = float('inf')
        for fx, fy in frontier:
            distance = np.sqrt((pos[0] - fx)**2 + (pos[1] - fy)**2)
            min_distance = min(min_distance, distance)
        
        # 靠近边界给予奖励
        distance_reward = max(0, 0.01 * (1.0 - min_distance / 10.0))
        
        return distance_reward
    
    def get_curiosity_driven_reward(self, current_features: Dict, previous_features: Dict = None) -> float:
       
        if previous_features is None:
            return 0
        
        # 计算关键特征的变化
        feature_changes = []
        
        # 等级变化
        level_change = abs(
            current_features.get("max_opponent_level", 0) - 
            previous_features.get("max_opponent_level", 0)
        )
        feature_changes.append(min(level_change / 10, 1.0) * 0.02)
        
        # 捕捉的Pokemon变化
        caught_change = abs(
            current_features.get("pokemon_caught", 0) - 
            previous_features.get("pokemon_caught", 0)
        )
        feature_changes.append(caught_change * 0.05)
        
        # 徽章变化
        badge_change = abs(
            current_features.get("badges_earned", 0) - 
            previous_features.get("badges_earned", 0)
        )
        feature_changes.append(badge_change * 0.5)
        
        curiosity_reward = sum(feature_changes)
        
        return min(curiosity_reward, 0.1)  # 限制上限
    
    def _find_frontier_tiles(self) -> Set[Tuple[int, int]]:
       
        frontier = set()
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if (x, y) not in self.visit_counts:
                    # 检查是否在已访问的磁贴附近
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if (x + dx, y + dy) in self.visit_counts:
                                frontier.add((x, y))
                                break
        
        return frontier
    
    def get_coverage_percentage(self) -> float:
       
        return (self.unique_tiles_visited / (self.grid_width * self.grid_height)) * 100
    
    def get_statistics(self) -> Dict:
       
        return {
            "unique_tiles_visited": self.unique_tiles_visited,
            "total_visits": self.total_visits,
            "coverage_percentage": self.get_coverage_percentage(),
            "frontier_size": len(self._find_frontier_tiles()),
            "visited_map": dict(self.visit_counts),
        }


class CuriosityDrivenExploration:
   
    
    def __init__(self, feature_dim: int = 32, hidden_dim: int = 128):
     
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 前向模型参数
        self.forward_model_loss = []
        self.prediction_errors = []
        
        # 好奇心历史
        self.curiosity_history = []
        self.max_history = 100
    
    def compute_prediction_error(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        predicted_next_state: np.ndarray
    ) -> float:
        
        # 计算欧几里得距离
        error = np.sqrt(np.mean((next_state - predicted_next_state) ** 2))
        
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.max_history:
            self.prediction_errors.pop(0)
        
        return error
    
    def get_intrinsic_reward(self, prediction_error: float) -> float:
       
        if len(self.prediction_errors) < 10:
            return prediction_error
        
        # 计算预测误差的统计
        errors = np.array(self.prediction_errors)
        error_mean = np.mean(errors)
        error_std = np.std(errors) + 1e-8
        
        # 标准化误差
        normalized_error = (prediction_error - error_mean) / error_std
        
        # 内在奖励（归一化）
        intrinsic_reward = np.tanh(normalized_error / 2.0) * 0.01
        
        self.curiosity_history.append(intrinsic_reward)
        if len(self.curiosity_history) > self.max_history:
            self.curiosity_history.pop(0)
        
        return intrinsic_reward
    
    def get_curiosity_statistics(self) -> Dict:
        if len(self.prediction_errors) == 0:
            return {
                "mean_prediction_error": 0,
                "std_prediction_error": 0,
                "mean_intrinsic_reward": 0,
            }
        
        errors = np.array(self.prediction_errors)
        rewards = np.array(self.curiosity_history)
        
        return {
            "mean_prediction_error": float(np.mean(errors)),
            "std_prediction_error": float(np.std(errors)),
            "max_prediction_error": float(np.max(errors)),
            "mean_intrinsic_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0,
        }


# ============= 使用示例 =============

if __name__ == "__main__":
    print("="*60)
    print("自适应探索奖励系统演示")
    print("="*60)
    
    # 初始化探索奖励系统
    explorer = AdaptiveExplorationReward(grid_width=20, grid_height=18)
    
    # 模拟探索过程
    print("\n模拟探索过程...")
    positions = [
        (2, 1), (3, 1), (4, 1), (5, 1),  # 向右移动
        (5, 2), (5, 3), (5, 4),  # 向下移动
        (4, 4), (3, 4), (2, 4),  # 向左移动
    ]
    
    for step, pos in enumerate(positions * 5):  # 重复5次
        reward = explorer.get_exploration_reward(pos, current_map=1)
        print(f"步骤 {step}: 位置 {pos}, 探索奖励: {reward:.4f}")
    
    # 输出统计
    stats = explorer.get_statistics()
    print(f"\n统计信息:")
    print(f"  访问过的唯一磁贴: {stats['unique_tiles_visited']}")
    print(f"  总访问次数: {stats['total_visits']}")
    print(f"  覆盖百分比: {stats['coverage_percentage']:.2f}%")
    print(f"  边界磁贴数: {stats['frontier_size']}")
    
    # 演示好奇心驱动探索
    print("\n" + "="*60)
    print("好奇心驱动探索演示")
    print("="*60)
    
    curiosity = CuriosityDrivenExploration(feature_dim=32)
    
    # 模拟状态转移
    for i in range(50):
        state = np.random.randn(32)
        next_state = state + np.random.randn(32) * 0.5
        predicted_next_state = state + np.random.randn(32) * 0.5
        
        error = curiosity.compute_prediction_error(
            state, action=0, next_state=next_state,
            predicted_next_state=predicted_next_state
        )
        reward = curiosity.get_intrinsic_reward(error)
        
        if (i + 1) % 10 == 0:
            print(f"步骤 {i+1}: 预测误差: {error:.4f}, 内在奖励: {reward:.6f}")
    
    # 输出好奇心统计
    curiosity_stats = curiosity.get_curiosity_statistics()
    print(f"\n好奇心统计:")
    print(f"  平均预测误差: {curiosity_stats['mean_prediction_error']:.4f}")
    print(f"  标准差: {curiosity_stats['std_prediction_error']:.4f}")
    print(f"  平均内在奖励: {curiosity_stats['mean_intrinsic_reward']:.6f}")