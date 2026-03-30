# -*- coding: utf-8 -*-
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
