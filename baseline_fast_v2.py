import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import sys
from os.path import exists
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from timeout_wrapper import TimeoutWrapper
import gymnasium as gym
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

try:
    from optimizations import (
        ImprovedRewardCalculator,
        AdaptiveExplorationReward,
        BattleRewardCalculator,
        PrioritizedExperienceBuffer,
        AdaptiveHyperparameterScheduler,
        AdvancedPrioritizedReplay,
        CuriosityDrivenExploration
    )

    HAS_OPTIMIZATIONS = True
except ImportError:
    HAS_OPTIMIZATIONS = False
    print("⚠️ 优化模块不可用")


# ✅ 修复：添加所有需要的方法
class GymnasiumWrapper(gym.Env):
    """完整的 gymnasium 兼容性包装器"""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # ✅ 初始化TensorBoard回调需要的属性
        self.agent_stats = []
        self.recent_screens = []
        self.explore_map = None
        self.current_event_flags_set = {}

    def _get_nested_attr(self, name):
        """从嵌套的环境中获取属性"""
        # 尝试不同层级的环境
        envs_to_check = [self.env]

        # 如果是包装器，检查底层环境
        if hasattr(self.env, 'env'):
            envs_to_check.append(self.env.env)
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'env'):
            envs_to_check.append(self.env.env.env)

        for env in envs_to_check:
            if hasattr(env, name):
                return getattr(env, name)
        return None

    # ✅ 添加 TensorBoard 回调需要的方法
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
        if stats is not None:
            return stats
        return self.agent_stats

    def get_recent_screens(self):
        """获取最近的屏幕截图"""
        screens = self._get_nested_attr('recent_screens')
        if screens is not None:
            return screens
        return self.recent_screens

    def get_explore_map(self):
        """获取探索地图"""
        explore_map = self._get_nested_attr('explore_map')
        if explore_map is not None:
            return explore_map

        # 如果没有，返回默认的空地图
        if self.explore_map is None:
            self.explore_map = np.zeros((144, 160), dtype=np.uint8)
        return self.explore_map

    def get_current_event_flags_set(self):
        """获取当前事件标志"""
        flags = self._get_nested_attr('current_event_flags_set')
        if flags is not None:
            return flags
        return self.current_event_flags_set

    def reset(self, seed=None, **kwargs):
        if hasattr(self.env, 'seed') and seed is not None:
            self.env.seed(seed)

        # ✅ 重置统计信息
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

        # ✅ 更新统计信息（如果可用的话）
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
        # 尝试从底层环境获取最新的统计信息
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
        # ✅ 特殊处理 TensorBoard 回调需要的属性
        if name == 'agent_stats':
            return self.get_agent_stats()
        elif name == 'recent_screens':
            return self.get_recent_screens()
        elif name == 'explore_map':
            return self.get_explore_map()
        elif name == 'current_event_flags_set':
            return self.get_current_event_flags_set()

        # ✅ 尝试从底层环境获取
        attr = self._get_nested_attr(name)
        if attr is not None:
            return attr

        # ✅ 最后尝试从直接环境获取
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
                    "user": "v2-default",
                    "env_id": rank,
                    "color": "#447799",
                    "extra": "",
                }
            )

            # 添加超时包装器
            env = TimeoutWrapper(env, timeout=15)

            # ✅ 添加 gymnasium 兼容性包装器
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


if __name__ == "__main__":
    use_wandb_logging = False
    ep_length = 2048
    sess_id = "runs"
    sess_path = Path(sess_id)

    # ============= 初始化优化器 =============
    if HAS_OPTIMIZATIONS:
        print("✅ 初始化优化奖励计算器...")
        reward_calculator = ImprovedRewardCalculator(
            reward_names=["exploration", "battle", "progress"],
            initial_weights={
                "exploration": 0.25,
                "battle": 0.25,
                "progress": 0.5,
            }
        )

        print("✅ 初始化自适应探索奖励...")
        exploration_reward = AdaptiveExplorationReward()

        print("✅ 初始化战斗奖励计算器...")
        battle_calculator = BattleRewardCalculator()

        print("✅ 初始化优先级经验回放...")
        prioritized_buffer = PrioritizedExperienceBuffer(max_size=100000, alpha=0.6)

        print("✅ 初始化超参数调度器...")
        scheduler = AdaptiveHyperparameterScheduler(initial_lr=3e-4, initial_exploration=0.2)

        print("✅ 初始化多任务目标...")
        multi_objectives = {
            "exploration_efficiency": {"weight": 0.3, "target": "maximize_visited_states"},
            "battle_performance": {"weight": 0.4, "target": "maximize_win_rate"},
            "map_coverage": {"weight": 0.3, "target": "maximize_coverage"},
        }

        print("✅ 初始化高级优先级回放...")
        advanced_replay = AdvancedPrioritizedReplay(max_size=100000)

        print("✅ 初始化好奇心驱动探索...")
        curiosity = CuriosityDrivenExploration()

        print("✅ 所有优化模块初始化完成！")
    else:
        reward_calculator = None
        exploration_reward = None
        battle_calculator = None
        prioritized_buffer = None
        scheduler = None
        multi_objectives = None
        advanced_replay = None
        curiosity = None

    env_config = {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': r'D:\PyCharm 2025.1\code_bailin\PokemonRedExperiments-master\PokemonRedExperiments-master\v5\manual_play_progress_GRASS_BEFORE_BATTLE.state',
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25,
        # ============= 优化配置 =============
        'reward_calculator': reward_calculator,
        'use_optimized_rewards': HAS_OPTIMIZATIONS,
        'exploration_reward': exploration_reward,
        'use_exploration_optimization': HAS_OPTIMIZATIONS,
        'battle_calculator': battle_calculator,
        'use_battle_optimization': HAS_OPTIMIZATIONS,
        'prioritized_buffer': prioritized_buffer,
        'use_prioritized_replay': HAS_OPTIMIZATIONS,
        'scheduler': scheduler,
        'use_scheduler': HAS_OPTIMIZATIONS,
        'multi_objectives': multi_objectives,
        'use_multi_task': HAS_OPTIMIZATIONS,
        'advanced_replay': advanced_replay,
        'use_advanced_replay': HAS_OPTIMIZATIONS,
        'curiosity': curiosity,
        'use_curiosity': HAS_OPTIMIZATIONS,
    }

    num_cpu = 6

    try:
        env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
        print(f"✅ 使用 {num_cpu} 个进程的多进程环境")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        exit(1)

    checkpoint_callback = CheckpointCallback(save_freq=ep_length // 2, save_path=sess_path,
                                             name_prefix="poke")

    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # 检查文件名
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()

    train_steps_batch = ep_length // 32

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env, device="cpu")
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device="cpu",
            tensorboard_log="runs",
            n_steps=train_steps_batch,
            batch_size=256,
            n_epochs=4,
            gamma=0.997,
            ent_coef=0.01,
        )

    print(model.policy)

    # 动态学习率设置
    if HAS_OPTIMIZATIONS and scheduler:
        total_timesteps = ep_length * num_cpu * 600
        initial_lr = scheduler.get_lr(total_timesteps, 0)
        model.learning_rate = initial_lr
        print(f"✅ 设置初始学习率: {initial_lr:.6f}")

    total_updates = 600
    try:
        model.learn(
            total_timesteps=ep_length * num_cpu * total_updates,
            callback=CallbackList(callbacks),
            tb_log_name="poke_ppo"
        )
    except Exception as e:
        print(f"训练过程中出错: {e}")
        print("尝试保存模型...")
        try:
            model.save(sess_path / "emergency_save")
            print("✅ 紧急保存完成")
        except Exception as save_error:
            print(f"❌ 紧急保存失败: {save_error}")
    finally:
        try:
            env.close()
        except:
            pass

    if use_wandb_logging:
        run.finish()



