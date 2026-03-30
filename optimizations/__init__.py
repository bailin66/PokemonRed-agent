"""Pokemon Red V2 优化模块"""
# ← 新增：高级优化类导出
from .advanced_optimization import (
    PrioritizedExperienceBuffer,
    AdaptiveHyperparameterScheduler,
    AdvancedPrioritizedReplay,
    CuriosityDrivenExploration,
)
# 基础优化
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

# 高级优化（可选）
try:
    from .network_optimization import ImprovedPolicyNetwork
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
