# -*- coding: utf-8 -*-
"""
高级优化模块 - 包含问题6、7、9、10的实现
"""

import numpy as np


# ============= 问题6：优先级经验回放 =============

class PrioritizedExperienceBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, max_size=100000, alpha=0.6):
        """
        Args:
            max_size: 缓冲区最大大小
            alpha: 优先级幂次（0=均匀，1=完全优先级）
        """
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
    
    def add(self, experience, td_error):
        """
        添加经验到缓冲区
        
        Args:
            experience: (state, action, reward, next_state, done)
            td_error: TD误差（用于计算优先级）
        """
        # 计算优先级：TD误差越大，优先级越高
        priority = (1.0 + abs(td_error)) ** self.alpha
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # 如果缓冲区满，替换优先级最低的
            if self.priorities:
                min_idx = self.priorities.index(min(self.priorities))
                self.buffer[min_idx] = experience
                self.priorities[min_idx] = priority
    
    def sample(self, batch_size):
        """
        根据优先级采样经验
        
        Returns:
            优先级采样的经验批次
        """
        if len(self.buffer) == 0:
            return []
        
        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / (priorities.sum() + 1e-8)
        
        # 根据概率采样
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probabilities,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]
    
    def update_priority(self, idx, td_error):
        """更新经验的优先级"""
        if idx < len(self.priorities):
            self.priorities[idx] = (1.0 + abs(td_error)) ** self.alpha


# ============= 问题7：自适应超参数调度器 =============

class AdaptiveHyperparameterScheduler:
    """自动调整超参数的调度器"""
    
    def __init__(self, initial_lr=3e-4, initial_exploration=0.2):
        """
        Args:
            initial_lr: 初始学习率
            initial_exploration: 初始探索系数
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.initial_exploration = initial_exploration
        self.current_exploration = initial_exploration
        self.step = 0
        self.total_steps = 0
    
    def get_lr(self, total_timesteps, current_timestep):
        """
        使用余弦退火调整学习率
        
        早期：学习率较高（快速学习）
        后期：学习率较低（精细调整）
        """
        progress = current_timestep / max(total_timesteps, 1)
        
        # 余弦退火公式
        lr = self.initial_lr * (1 + np.cos(np.pi * progress)) / 2
        self.current_lr = lr
        return lr
    
    def get_exploration_rate(self, total_timesteps, current_timestep):
        """
        动态调整探索率
        
        早期：探索多（新奇度奖励高）
        后期：利用多（减少探索）
        """
        progress = current_timestep / max(total_timesteps, 1)
        
        # 线性衰减
        exploration = self.initial_exploration * (1 - progress * 0.5)
        self.current_exploration = max(exploration, 0.05)
        return self.current_exploration
    
    def step_callback(self, total_timesteps, current_timestep):
        """在训练过程中调用"""
        lr = self.get_lr(total_timesteps, current_timestep)
        exploration = self.get_exploration_rate(total_timesteps, current_timestep)
        
        return {
            'learning_rate': lr,
            'exploration_rate': exploration,
            'progress': current_timestep / max(total_timesteps, 1)
        }


# ============= 问题9：优先级经验回放进阶 =============

class AdvancedPrioritizedReplay:
    """高级优先级回放（考虑TD误差和新奇度）"""
    
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
        self.td_errors = []
        self.novelty_scores = []
    
    def add(self, experience, td_error, novelty_score=1.0):
        """
        添加经验，同时考虑TD误差和新奇度
        
        优先级 = (TD误差 + 新奇度) / 2
        """
        # 结合TD误差和新奇度
        combined_priority = 0.5 * (1.0 + abs(td_error)) + 0.5 * novelty_score
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(combined_priority)
            self.td_errors.append(td_error)
            self.novelty_scores.append(novelty_score)
        else:
            # 替换优先级最低的
            if self.priorities:
                min_idx = self.priorities.index(min(self.priorities))
                self.buffer[min_idx] = experience
                self.priorities[min_idx] = combined_priority
                self.td_errors[min_idx] = td_error
                self.novelty_scores[min_idx] = novelty_score
    
    def get_batch_weights(self, batch_size):
        """获取采样权重（用于加权学习）"""
        if len(self.buffer) == 0:
            return [], []
        
        priorities = np.array(self.priorities, dtype=np.float32)
        weights = priorities / (priorities.sum() + 1e-8)
        
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=weights,
            replace=False
        )
        
        return indices, weights[indices]


# ============= 问题10：好奇心驱动探索 =============

class CuriosityDrivenExploration:
    """基于好奇心的探索模块"""
    
    def __init__(self, curiosity_scale=0.1, feature_dim=128):
        """
        Args:
            curiosity_scale: 好奇心强度（0-1）
            feature_dim: 特征维度
        """
        self.curiosity_scale = curiosity_scale
        self.feature_dim = feature_dim
        self.state_predictions = {}  # 预测误差缓存
        self.exploration_count = {}
    
    def calculate_curiosity(self, state_hash, prediction_error):
        """
        计算好奇心奖励
        
        预测误差大 = 模型对这个状态不理解 = 应该探索
        """
        # 基础好奇心：预测误差
        curiosity = min(prediction_error * self.curiosity_scale, 1.0)
        
        # 加成：这个状态很少访问
        visit_count = self.exploration_count.get(state_hash, 0)
        visit_bonus = 1.0 / (1.0 + visit_count)
        
        # 总好奇心 = 预测误差 * 访问频率
        total_curiosity = curiosity * visit_bonus
        
        # 更新访问计数
        self.exploration_count[state_hash] = visit_count + 1
        
        return total_curiosity
    
    def update_prediction(self, state_hash, prediction_error):
        """更新对某个状态的预测误差"""
        self.state_predictions[state_hash] = prediction_error
    
    def get_curiosity_reward(self, state_hash):
        """获取好奇心奖励"""
        prediction_error = self.state_predictions.get(state_hash, 0.5)
        return self.calculate_curiosity(state_hash, prediction_error)
