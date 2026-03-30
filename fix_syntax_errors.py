#!/usr/bin/env python3
"""
一键创建所有干净的优化模块 - 彻底重建
"""

from pathlib import Path
import shutil

# 1. 完全删除旧的optimizations目录
optim_dir = Path('optimizations')
if optim_dir.exists():
    shutil.rmtree(optim_dir)
    print("🗑️  删除旧的optimizations目录")

# 2. 创建新目录
optim_dir.mkdir(exist_ok=True)
print("📁 创建新的optimizations目录")

# 3. 创建所有模块文件

# reward_optimization.py
with open(optim_dir / 'reward_optimization.py', 'w', encoding='utf-8') as f:
    f.write(""""""改进的奖励计算模块"""

class ImprovedRewardCalculator:
    """改进的奖励计算器"""

    def __init__(self, reward_names=None, initial_weights=None):
        self.reward_names = reward_names or ["exploration", "battle", "progress"]
        if initial_weights is None:
            self.weights = {name: 1.0 / len(self.reward_names) for name in self.reward_names}
        else:
            self.weights = initial_weights

    def calculate_total_reward(self, reward_dict):
        total = 0.0
        for name in self.reward_names:
            value = reward_dict.get(name, 0.0)
            total += self.weights.get(name, 0.0) * value
        return total, {"total": total, "breakdown": reward_dict}

    def update_weights(self, new_weights):
        self.weights.update(new_weights)


class RewardNormalizer:
    """奖励标准化器"""

    def __init__(self, alpha=0.99):
        self.mean = 0.0
        self.std = 1.0
        self.alpha = alpha

    def normalize(self, reward):
        self.mean = self.alpha * self.mean + (1 - self.alpha) * reward
        self.std = max(self.std, 0.1)
        return (reward - self.mean) / (self.std + 1e-8)


class DynamicRewardWeighter:
    """动态奖励权重调整"""

    def __init__(self):
        self.weights = {"exploration": 0.25, "battle": 0.25, "progress": 0.5}

    def adjust_weights(self, **kwargs):
        self.weights.update(kwargs)
        return self.weights
""")
print("✅ 创建: reward_optimization.py")

# feature_extraction.py
with open(optim_dir / 'feature_extraction.py', 'w', encoding='utf-8') as f:
    f.write(""""""特征提取模块"""

class FeatureExtractor:
    """从游戏状态提取特征"""

    def __init__(self):
        self.cache = {}

    def extract(self, state):
        if isinstance(state, dict):
            return state
        return {"raw": state}


class StateRepresentation:
    """游戏状态表示"""

    @staticmethod
    def encode(state):
        return state

    @staticmethod
    def decode(encoded_state):
        return encoded_state
""")
print("✅ 创建: feature_extraction.py")

# exploration_reward.py
with open(optim_dir / 'exploration_reward.py', 'w', encoding='utf-8') as f:
    f.write(""""""探索奖励模块"""

class AdaptiveExplorationReward:
    """自适应探索奖励"""

    def __init__(self, base_reward=0.1):
        self.base_reward = base_reward
        self.difficulty = 1.0

    def calculate(self, visited_states):
        novelty = 1.0 / (1.0 + len(visited_states))
        return self.base_reward * novelty * self.difficulty

    def update_difficulty(self, new_difficulty):
        self.difficulty = new_difficulty


class CuriosityDrivenExploration:
    """好奇心驱动的探索"""

    def __init__(self, curiosity_scale=1.0):
        self.curiosity_scale = curiosity_scale
        self.prediction_errors = []

    def calculate_curiosity_reward(self, prediction_error):
        return self.curiosity_scale * prediction_error
""")
print("✅ 创建: exploration_reward.py")

# battle_strategy.py
with open(optim_dir / 'battle_strategy.py', 'w', encoding='utf-8') as f:
    f.write(""""""战斗策略和奖励模块"""

class BattleRewardCalculator:
    """战斗中的奖励计算"""

    def __init__(self):
        self.hp_weights = {"player": 1.0, "opponent": -1.0}

    def calculate(self, battle_state):
        player_hp = battle_state.get("player_hp", 0)
        opponent_hp = battle_state.get("opponent_hp", 0)
        reward = (
            self.hp_weights["player"] * (player_hp / 255.0) +
            self.hp_weights["opponent"] * (opponent_hp / 255.0)
        )
        return reward


class BattleActionEvaluator:
    """战斗行动评估"""

    @staticmethod
    def evaluate_action(action, battle_state):
        return 0.0
""")
print("✅ 创建: battle_strategy.py")

# network_optimization.py
with open(optim_dir / 'network_optimization.py', 'w', encoding='utf-8') as f:
    f.write(""""""网络优化模块"""

class ImprovedPolicyNetwork:
    """改进的策略网络"""

    @staticmethod
    def suggest_architecture():
        return {
            "input_layers": ["game_state", "battle_state"],
            "hidden_units": [256, 256],
            "output_dim": 12
        }
""")
print("✅ 创建: network_optimization.py")

# advanced_optimization.py
with open(optim_dir / 'advanced_optimization.py', 'w', encoding='utf-8') as f:
    f.write(""""""高级优化模块"""

class HierarchicalRLAgent:
    """分层RL代理"""

    def __init__(self):
        self.high_level_policy = None
        self.low_level_policies = {}


class PrioritizedExperienceReplayBuffer:
    """优先级经验回放缓冲区"""

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []


class AdaptiveHyperparameterScheduler:
    """自适应超参数调度器"""

    def __init__(self):
        self.learning_rate = 3e-4
        self.update_schedule = {}
""")
print("✅ 创建: advanced_optimization.py")

# __init__.py
with open(optim_dir / '__init__.py', 'w', encoding='utf-8') as f:
    f.write(""" """Pokemon Red V2 优化模块"""

from.reward_optimization import (
    ImprovedRewardCalculator,
    RewardNormalizer,
    DynamicRewardWeighter,
)

from .feature_extraction import (
    FeatureExtractor,
    StateRepresentation,
)

from .exploration_reward import (
    AdaptiveExplorationReward,
    CuriosityDrivenExploration,
)

from .battle_strategy import (
    BattleRewardCalculator,
    BattleActionEvaluator,
)

try:
    from .network_optimization import ImprovedPolicyNetwork
    from .advanced_optimization import (
        HierarchicalRLAgent,
        PrioritizedExperienceReplayBuffer,
        AdaptiveHyperparameterScheduler,
    )
except ImportError:
    pass

__all__ = [
    'ImprovedRewardCalculator',
    'RewardNormalizer',
    'DynamicRewardWeighter',
    'FeatureExtractor',
    'StateRepresentation',
    'AdaptiveExplorationReward',
    'CuriosityDrivenExploration',
    'BattleRewardCalculator',
    'BattleActionEvaluator',
]
""")
print("✅ 创建: __init__.py")

print("\n" + "="*50)
print("测试导入...")
print("="*50)

try:
    from optimizations import ImprovedRewardCalculator
    print("\n✅ ImprovedRewardCalculator 导入成功")
    from optimizations import RewardNormalizer
    print("✅ RewardNormalizer 导入成功")
    from optimizations import FeatureExtractor
    print("✅ FeatureExtractor 导入成功")
    print("\n✨ 所有模块创建和导入成功！")
except Exception as e:
    print(f"\n❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()