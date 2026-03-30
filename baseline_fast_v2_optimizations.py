"""
Pokemon Red 微调训练脚本
基于已有的检查点进行继续训练，使用优化的奖励函数
"""

import os
import time
import warnings
from pathlib import Path
import numpy as np
import sys

# 设置环境变量
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# 导入必要的库
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from timeout_wrapper import TimeoutWrapper
import gymnasium as gym

# 忽略警告
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# 尝试导入优化模块
try:
    from optimizations import (
        ImprovedRewardCalculator,
        AdaptiveExplorationReward,
        BattleRewardCalculator,
        PrioritizedExperienceBuffer,
        AdaptiveHyperparameterScheduler,
        AdvancedPrioritizedReplay,
        CuriosityDrivenExploration,
    )
    HAS_OPTIMIZATIONS = True
    print("✅ 优化模块已加载")
except ImportError:
    HAS_OPTIMIZATIONS = False
    print("⚠️ 优化模块不可用，使用基础训练")


class GymnasiumWrapper(gym.Env):
    """完整的 gymnasium 兼容性包装器"""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # 初始化TensorBoard回调需要的属性
        self.agent_stats = []
        self.recent_screens = []
        self.explore_map = None
        self.current_event_flags_set = {}

    def _get_nested_attr(self, name):
        """从嵌套的环境中获取属性"""
        envs_to_check = [self.env]

        if hasattr(self.env, 'env'):
            envs_to_check.append(self.env.env)
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'env'):
            envs_to_check.append(self.env.env.env)

        for env in envs_to_check:
            if hasattr(env, name):
                return getattr(env, name)
        return None

    def check_if_done(self):
        """检查环境是否完成"""
        method = self._get_nested_attr('check_if_done')
        if method and callable(method):
            try:
                return method()
            except:
                return False
        return False

    def get_agent_stats(self):
        """获取智能体统计信息"""
        stats = self._get_nested_attr('agent_stats')
        return stats if stats is not None else self.agent_stats

    def get_recent_screens(self):
        """获取最近的屏幕截图"""
        screens = self._get_nested_attr('recent_screens')
        return screens if screens is not None else self.recent_screens

    def get_explore_map(self):
        """获取探索地图"""
        explore_map = self._get_nested_attr('explore_map')
        if explore_map is not None:
            return explore_map

        if self.explore_map is None:
            self.explore_map = np.zeros((144, 160), dtype=np.uint8)
        return self.explore_map

    def get_current_event_flags_set(self):
        """获取当前事件标志"""
        flags = self._get_nested_attr('current_event_flags_set')
        return flags if flags is not None else self.current_event_flags_set

    def reset(self, seed=None, **kwargs):
        if hasattr(self.env, 'seed') and seed is not None:
            self.env.seed(seed)

        # 重置统计信息
        self.agent_stats = []
        self.recent_screens = []
        self.current_event_flags_set = {}

        try:
            obs = self.env.reset(seed=seed, **kwargs)
        except TypeError:
            obs = self.env.reset()

        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        result = self.env.step(action)

        # 更新统计信息
        try:
            self._update_stats()
        except:
            pass

        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        elif len(result) == 5:
            return result
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} values")

    def _update_stats(self):
        """更新统计信息"""
        agent_stats = self._get_nested_attr('agent_stats')
        if agent_stats is not None:
            self.agent_stats = agent_stats

        recent_screens = self._get_nested_attr('recent_screens')
        if recent_screens is not None:
            self.recent_screens = recent_screens

        explore_map = self._get_nested_attr('explore_map')
        if explore_map is not None:
            self.explore_map = explore_map

        flags = self._get_nested_attr('current_event_flags_set')
        if flags is not None:
            self.current_event_flags_set = flags

    def render(self, mode='human'):
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()

    def __getattr__(self, name):
        """传递所有其他属性和方法"""
        if name == 'agent_stats':
            return self.get_agent_stats()
        elif name == 'recent_screens':
            return self.get_recent_screens()
        elif name == 'explore_map':
            return self.get_explore_map()
        elif name == 'current_event_flags_set':
            return self.get_current_event_flags_set()

        attr = self._get_nested_attr(name)
        if attr is not None:
            return attr

        return getattr(self.env, name)


def make_env(rank, env_conf, seed=0):
    """创建环境的工厂函数"""
    def _init():
        try:
            # 创建基础环境
            base_env = RedGymEnv(env_conf)

            # 包装为流式环境
            env = StreamWrapper(
                base_env,
                stream_metadata={
                    "user": f"finetune-v{rank}",
                    "env_id": rank,
                    "color": "#ff6b6b",  # 红色表示微调
                    "extra": "finetune",
                }
            )

            # 添加超时包装器
            env = TimeoutWrapper(env, timeout=15)

            # 添加 gymnasium 兼容性包装器
            env = GymnasiumWrapper(env)

            try:
                env.reset(seed=(seed + rank))
            except Exception as e:
                print(f"Warning: Reset failed for env {rank}: {e}")
                env.reset()

            return env

        except Exception as e:
            print(f"Error creating environment {rank}: {e}")
            raise

    set_random_seed(seed)
    return _init


def load_and_validate_checkpoint(checkpoint_path):
    """加载并验证检查点文件"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return None

    try:
        # 先尝试加载检查基本信息
        print(f"🔍 验证检查点: {checkpoint_path}")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.2f} MB")

        return checkpoint_path

    except Exception as e:
        print(f"❌ 检查点验证失败: {e}")
        return None


def setup_finetune_config():
    """设置微调专用配置"""

    # =================== 🎯 核心微调设置 ===================
    config = {
        # 检查点设置
        "CHECKPOINT_PATH": r"runs\poke_ppo_17\poke_2334720_steps.zip",  # 🔧 修改为你的模型路径
        "FINETUNE_LEARNING_RATE": 0.0003,

        # 训练设置
        "EPISODE_LENGTH": 4096,
        "NUM_PROCESSES": 6,
        "TOTAL_UPDATES": 600,

        # 环境设置
        "USE_WANDB": False,
        "SAVE_FREQ": 4096,

        # 奖励优化设置
        "REWARD_SCALE": 4.0,      # 增强奖励信号
        "EXPLORE_WEIGHT": 2.5,    # 提高探索奖励
        "USE_DEATH_PENALTY": True,
        "DEATH_PENALTY": -2.0,
        "USE_BATTLE_REWARDS": True,
    }

    return config


def initialize_optimizations():
    """初始化优化模块"""
    optimizations = {}

    if not HAS_OPTIMIZATIONS:
        print("⚠️ 跳过优化模块初始化")
        return optimizations

    try:
        print("🚀 初始化微调专用优化器...")

        # 改进的奖励计算器
        optimizations['reward_calculator'] = ImprovedRewardCalculator(
            reward_names=["exploration", "battle", "progress", "efficiency"],
            initial_weights={
                "exploration": 0.2,  # 微调时降低探索权重
                "battle": 0.4,       # 增加战斗权重
                "progress": 0.3,     # 增加进度权重
                "efficiency": 0.1,   # 新增效率权重
            }
        )

        # 自适应探索奖励
        optimizations['exploration_reward'] = AdaptiveExplorationReward()

        # 战斗奖励计算器
        optimizations['battle_calculator'] = BattleRewardCalculator()

        # 优先级经验回放
        optimizations['prioritized_buffer'] = PrioritizedExperienceBuffer(
            max_size=50000,  # 微调时使用较小缓冲区
            alpha=0.7       # 增加优先级强度
        )

        # 超参数调度器
        optimizations['scheduler'] = AdaptiveHyperparameterScheduler(
            initial_lr=0.0003,  # 微调专用学习率
            initial_exploration=0.1  # 降低初始探索
        )

        # 高级优先级回放
        optimizations['advanced_replay'] = AdvancedPrioritizedReplay(max_size=50000)

        # 好奇心驱动探索
        optimizations['curiosity'] = CuriosityDrivenExploration()

        print("✅ 所有优化模块初始化完成！")

    except Exception as e:
        print(f"⚠️ 优化模块初始化部分失败: {e}")

    return optimizations


def create_finetune_env_config(config, optimizations):
    """创建微调专用环境配置"""

    # 创建会话路径
    timestamp = int(time.time())
    sess_path = Path(f"finetune_session_{timestamp}")
    sess_path.mkdir(exist_ok=True)

    env_config = {
        # 基础环境设置
        'headless': True,
        'save_final_state': True,  # 微调时保存最终状态
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
        'max_steps': config["EPISODE_LENGTH"],
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,

        # ✅ 微调专用奖励设置
        'reward_scale': config["REWARD_SCALE"],
        'explore_weight': config["EXPLORE_WEIGHT"],

        # ✅ 新增奖励机制
        'use_death_penalty': config["USE_DEATH_PENALTY"],
        'death_penalty_scale': config["DEATH_PENALTY"],
        'use_battle_rewards': config["USE_BATTLE_REWARDS"],
        'use_dual_level_rewards': True,
        'player_level_weight': 0.4,
        'opponent_level_weight': 0.3,

        # ✅ 微调专用优化
        'use_adaptive_difficulty': True,
        'difficulty_adaptation_rate': 0.1,
        'use_progressive_rewards': True,
        'progress_milestone_bonus': 5.0,

        # 优化模块配置
        'reward_calculator': optimizations.get('reward_calculator'),
        'use_optimized_rewards': HAS_OPTIMIZATIONS,
        'exploration_reward': optimizations.get('exploration_reward'),
        'use_exploration_optimization': HAS_OPTIMIZATIONS,
        'battle_calculator': optimizations.get('battle_calculator'),
        'use_battle_optimization': HAS_OPTIMIZATIONS,
        'prioritized_buffer': optimizations.get('prioritized_buffer'),
        'use_prioritized_replay': HAS_OPTIMIZATIONS,
        'scheduler': optimizations.get('scheduler'),
        'use_scheduler': HAS_OPTIMIZATIONS,
        'advanced_replay': optimizations.get('advanced_replay'),
        'use_advanced_replay': HAS_OPTIMIZATIONS,
        'curiosity': optimizations.get('curiosity'),
        'use_curiosity': HAS_OPTIMIZATIONS,
    }

    return env_config, sess_path


def main():
    """主微调函数"""
    print("🎯 Pokemon Red 微调训练开始!")
    print("=" * 50)

    # 1. 加载配置
    config = setup_finetune_config()
    print(f"📋 配置加载完成")

    # 2. 验证检查点
    checkpoint_path = load_and_validate_checkpoint(config["CHECKPOINT_PATH"])
    if checkpoint_path is None:
        print("❌ 无法加载检查点，退出训练")
        return

    print(f"✅ 检查点验证通过: {checkpoint_path}")

    # 3. 初始化优化模块
    optimizations = initialize_optimizations()

    # 4. 创建环境配置
    env_config, sess_path = create_finetune_env_config(config, optimizations)
    print(f"🌍 环境配置创建完成，会话路径: {sess_path}")

    # 5. 创建多进程环境
    num_cpu = config["NUM_PROCESSES"]
    print(f"🔧 创建 {num_cpu} 个并行环境...")

    try:
        env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
        print(f"✅ 多进程环境创建成功")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return

    # 6. 设置回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=config["SAVE_FREQ"],
        save_path=sess_path,
        name_prefix="finetune_poke"
    )

    callbacks = [
        checkpoint_callback,
        TensorboardCallback(sess_path)
    ]

    # 7. 加载模型进行微调
    print(f"🔄 加载预训练模型...")
    try:
        model = PPO.load(checkpoint_path, env=env, device="cpu")
        print(f"✅ 模型加载成功")

        # 调整微调参数
        model.learning_rate = config["FINETUNE_LEARNING_RATE"]
        print(f"📉 学习率: {model.learning_rate}")

        # 重新设置训练参数
        train_steps_batch = config["EPISODE_LENGTH"] // 32
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()

        print("🔧 训练参数重置完成")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 8. 开始微调训练
    total_timesteps = config["EPISODE_LENGTH"] * num_cpu * config["TOTAL_UPDATES"]
    print(f"🚀 开始微调训练...")
    print(f"📊 总训练步数: {total_timesteps:,}")
    print(f"📊 预计训练时长: {config['TOTAL_UPDATES']} 个更新周期")
    print("=" * 50)

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name="pokemon_finetune",
            reset_num_timesteps=False  # ⭐ 重要：不重置步数计数
        )

        # 训练完成，保存最终模型
        final_model_path = sess_path / "finetune_final_model"
        model.save(final_model_path)
        print(f"💾 最终模型已保存: {final_model_path}")

    except KeyboardInterrupt:
        print("\n⏸️ 训练被用户中断")
        emergency_path = sess_path / f"finetune_emergency_{int(time.time())}"
        model.save(emergency_path)
        print(f"💾 紧急保存完成: {emergency_path}")

    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        emergency_path = sess_path / f"finetune_error_{int(time.time())}"
        try:
            model.save(emergency_path)
            print(f"💾 错误恢复保存: {emergency_path}")
        except:
            print("❌ 无法保存模型")

    finally:
        # 清理资源
        try:
            env.close()
            print("🧹 环境资源清理完成")
        except:
            pass

        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ 总训练时间: {duration/3600:.2f} 小时")

    print("🎉 微调训练完成!")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"📝 使用命令行指定的检查点: {checkpoint_path}")
        # 这里可以修改配置中的检查点路径

    main()