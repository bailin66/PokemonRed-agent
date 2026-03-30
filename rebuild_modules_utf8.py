#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键创建所有干净的优化模块 - UTF-8编码修复版本
直接复制这个文件到v2目录并运行: python rebuild_modules_utf8.py
"""

from pathlib import Path
import shutil

def main():
    # 删除旧的optimizations目录
    optim_dir = Path('optimizations')
    if optim_dir.exists():
        shutil.rmtree(optim_dir)
        print("删除旧的optimizations目录")
    
    # 创建新目录
    optim_dir.mkdir(exist_ok=True)
    print("创建新的optimizations目录")
    print("")
    
    # 文件1: reward_optimization.py
    reward_code = """# -*- coding: utf-8 -*-
class ImprovedRewardCalculator:
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
    def __init__(self, alpha=0.99):
        self.mean = 0.0
        self.std = 1.0
        self.alpha = alpha
    
    def normalize(self, reward):
        self.mean = self.alpha * self.mean + (1 - self.alpha) * reward
        self.std = max(self.std, 0.1)
        return (reward - self.mean) / (self.std + 1e-8)


class DynamicRewardWeighter:
    def __init__(self):
        self.weights = {"exploration": 0.25, "battle": 0.25, "progress": 0.5}
    
    def adjust_weights(self, **kwargs):
        self.weights.update(kwargs)
        return self.weights
"""
    
    with open(optim_dir / 'reward_optimization.py', 'w', encoding='utf-8') as f:
        f.write(reward_code)
    print("✅ 创建: reward_optimization.py")
    
    # 文件2: feature_extraction.py
    feature_code = """# -*- coding: utf-8 -*-
class FeatureExtractor:
    def __init__(self):
        self.cache = {}
    
    def extract(self, state):
        if isinstance(state, dict):
            return state
        return {"raw": state}


class StateRepresentation:
    @staticmethod
    def encode(state):
        return state
    
    @staticmethod
    def decode(encoded_state):
        return encoded_state
"""
    
    with open(optim_dir / 'feature_extraction.py', 'w', encoding='utf-8') as f:
        f.write(feature_code)
    print("✅ 创建: feature_extraction.py")
    
    # 文件3: exploration_reward.py
    exploration_code = """# -*- coding: utf-8 -*-
class AdaptiveExplorationReward:
    def __init__(self, base_reward=0.1):
        self.base_reward = base_reward
        self.difficulty = 1.0
    
    def calculate(self, visited_states):
        novelty = 1.0 / (1.0 + len(visited_states))
        return self.base_reward * novelty * self.difficulty
    
    def update_difficulty(self, new_difficulty):
        self.difficulty = new_difficulty


class CuriosityDrivenExploration:
    def __init__(self, curiosity_scale=1.0):
        self.curiosity_scale = curiosity_scale
        self.prediction_errors = []
    
    def calculate_curiosity_reward(self, prediction_error):
        return self.curiosity_scale * prediction_error
"""
    
    with open(optim_dir / 'exploration_reward.py', 'w', encoding='utf-8') as f:
        f.write(exploration_code)
    print("✅ 创建: exploration_reward.py")
    
    # 文件4: battle_strategy.py
    battle_code = """# -*- coding: utf-8 -*-
class BattleRewardCalculator:
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
    @staticmethod
    def evaluate_action(action, battle_state):
        return 0.0
"""
    
    with open(optim_dir / 'battle_strategy.py', 'w', encoding='utf-8') as f:
        f.write(battle_code)
    print("✅ 创建: battle_strategy.py")
    
    # 文件5: network_optimization.py
    network_code = """# -*- coding: utf-8 -*-
class ImprovedPolicyNetwork:
    @staticmethod
    def suggest_architecture():
        return {
            "input_layers": ["game_state", "battle_state"],
            "hidden_units": [256, 256],
            "output_dim": 12
        }
"""
    
    with open(optim_dir / 'network_optimization.py', 'w', encoding='utf-8') as f:
        f.write(network_code)
    print("✅ 创建: network_optimization.py")
    
    # 文件6: advanced_optimization.py
    advanced_code = """# -*- coding: utf-8 -*-
class HierarchicalRLAgent:
    def __init__(self):
        self.high_level_policy = None
        self.low_level_policies = {}


class PrioritizedExperienceReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []


class AdaptiveHyperparameterScheduler:
    def __init__(self):
        self.learning_rate = 3e-4
        self.update_schedule = {}
"""
    
    with open(optim_dir / 'advanced_optimization.py', 'w', encoding='utf-8') as f:
        f.write(advanced_code)
    print("✅ 创建: advanced_optimization.py")
    
    # 文件7: __init__.py - 只使用ASCII，避免编码问题
    init_code = """# -*- coding: utf-8 -*-
from .reward_optimization import (
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
"""
    
    with open(optim_dir / '__init__.py', 'w', encoding='utf-8') as f:
        f.write(init_code)
    print("✅ 创建: __init__.py")
    
    # 测试导入
    print("\n" + "="*50)
    print("测试导入...")
    print("="*50 + "\n")
    
    try:
        from optimizations import ImprovedRewardCalculator
        print("✅ ImprovedRewardCalculator 导入成功")
        from optimizations import RewardNormalizer
        print("✅ RewardNormalizer 导入成功")
        from optimizations import FeatureExtractor
        print("✅ FeatureExtractor 导入成功")
        print("\n✨ 所有模块创建和导入成功！")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
