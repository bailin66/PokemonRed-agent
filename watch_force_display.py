import os
import gc  # ← 添加垃圾回收
import psutil  # ← 添加内存监控

os.environ['SDL_VIDEODRIVER'] = 'windib'

import sys
import time
from pathlib import Path
import warnings
import numpy as np
import pygame

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from timeout_wrapper import TimeoutWrapper
import gymnasium as gym


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0

    def get_memory_info(self):
        """获取内存使用信息"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, memory_mb)
            return {
                'current_mb': memory_mb,
                'peak_mb': self.peak_memory,
                'percent': self.process.memory_percent()
            }
        except:
            return {'current_mb': 0, 'peak_mb': 0, 'percent': 0}

    def should_clean(self, memory_mb):
        """判断是否需要清理内存"""
        return memory_mb > 1024  # 超过1GB就清理


class ForceDisplayWrapper:
    """优化的显示包装器 - 加入内存管理"""

    def __init__(self, env):
        self.env = env
        self.pygame_screen = None
        self.setup_pygame_display()
        self.last_display_update = 0
        self.update_counter = 0

    def setup_pygame_display(self):
        """设置pygame显示窗口"""
        try:
            pygame.init()
            self.pygame_screen = pygame.display.set_mode((640, 576))
            pygame.display.set_caption("Pokemon Red - AI Playing (内存优化版)")
            print("✅ 优化显示窗口创建成功")
        except Exception as e:
            print(f"❌ 强制显示窗口创建失败: {e}")

    def update_display(self):
        """优化的显示更新 - 限制频率并管理内存"""
        current_time = time.time()

        # 限制更新频率到30 FPS
        if current_time - self.last_display_update < 0.033:
            return

        self.last_display_update = current_time
        self.update_counter += 1

        if self.pygame_screen is None:
            return

        try:
            pyboy = self._get_pyboy()
            if pyboy is not None:
                try:
                    screen_pil = pyboy.screen_image()
                    screen_array = np.array(screen_pil)

                    if len(screen_array.shape) == 2:
                        screen_array = np.stack([screen_array] * 3, axis=2)
                    elif screen_array.shape[2] == 4:
                        screen_array = screen_array[:, :, :3]

                    surface = pygame.surfarray.make_surface(
                        np.transpose(screen_array, (1, 0, 2))
                    )

                    scaled_surface = pygame.transform.scale(surface, (640, 576))
                    self.pygame_screen.blit(scaled_surface, (0, 0))
                    pygame.display.flip()

                    # 快速处理事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pass

                    # 🔧 定期清理pygame表面
                    if self.update_counter % 300 == 0:
                        del surface, scaled_surface
                        gc.collect()

                except Exception as e:
                    print(f"⚠️ 显示更新异常: {e}")  # ← 不再静默处理
                    self.pygame_screen.fill((0, 0, 0))
                    pygame.display.flip()
            else:
                self.pygame_screen.fill((100, 0, 0))
                pygame.display.flip()

        except Exception as e:
            print(f"⚠️ 显示异常: {e}")  # ← 不再静默处理

    def _get_pyboy(self):
        """递归查找PyBoy实例"""
        if hasattr(self.env, 'pyboy'):
            return self.env.pyboy

        current_env = self.env
        for _ in range(10):
            if hasattr(current_env, 'env'):
                current_env = current_env.env
                if hasattr(current_env, 'pyboy'):
                    return current_env.pyboy
            else:
                break

        return None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.update_display()
        return result

    def step(self, action):
        result = self.env.step(action)
        self.update_display()
        return result

    def close(self):
        print("🔧 关闭显示组件...")
        if self.pygame_screen:
            pygame.quit()
        if hasattr(self.env, 'close'):
            self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class GymnasiumWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _get_nested_attr(self, name):
        envs_to_check = [self.env]
        if hasattr(self.env, 'env'):
            envs_to_check.append(self.env.env)
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'env'):
            envs_to_check.append(self.env.env.env)
        for env in envs_to_check:
            if hasattr(env, name):
                return getattr(env, name)
        return None

    def reset(self, seed=None, **kwargs):
        try:
            obs = self.env.reset(seed=seed, **kwargs)
        except TypeError:
            obs = self.env.reset()
        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()

    def __getattr__(self, name):
        attr = self._get_nested_attr(name)
        if attr is not None:
            return attr
        return getattr(self.env, name)


def make_watch_env(env_conf, rank=0):
    """创建观看环境"""
    base_env = RedGymEnv(env_conf)

    env = StreamWrapper(
        base_env,
        stream_metadata={
            "user": "v2-watch-stable",  # ← 修改用户名
            "env_id": rank,
            "color": "#447799",
            "extra": "memory-optimized",
        }
    )

    env = TimeoutWrapper(env, timeout=30)  # ← 增加超时时间
    env = ForceDisplayWrapper(env)
    env = GymnasiumWrapper(env)

    try:
        env.reset(seed=42 + rank)
    except Exception as e:
        print(f"⚠️ 环境重置异常: {e}")
        env.reset()

    return env


def safe_read_memory(env, address, default=0):
    """安全读取内存地址"""
    try:
        pyboy = None

        if hasattr(env, 'pyboy'):
            pyboy = env.pyboy

        if pyboy is None:
            current_env = env
            for _ in range(10):
                if hasattr(current_env, 'env'):
                    current_env = current_env.env
                    if hasattr(current_env, 'pyboy'):
                        pyboy = current_env.pyboy
                        break
                else:
                    break

        if pyboy is None:
            current_env = env
            for _ in range(10):
                if hasattr(current_env, 'read_m'):
                    return current_env.read_m(address)
                if hasattr(current_env, 'env'):
                    current_env = current_env.env
                else:
                    break

        if pyboy is not None:
            return pyboy.get_memory_value(address)

        return default

    except Exception as e:
        # print(f"读取内存地址 0x{address:X} 失败: {e}")  # 减少错误输出
        return default


def get_detailed_stats(env):
    """获取详细的游戏统计信息"""
    try:
        x_pos = safe_read_memory(env, 0xD362, -1)
        y_pos = safe_read_memory(env, 0xD361, -1)
        map_id = safe_read_memory(env, 0xD35E, -1)

        badges_byte = safe_read_memory(env, 0xD356, 0)
        badge_count = bin(badges_byte).count('1')

        try:
            current_hp = safe_read_memory(env, 0xD16C, 0) * 256 + safe_read_memory(env, 0xD16D, 0)
            max_hp = safe_read_memory(env, 0xD18D, 1) * 256 + safe_read_memory(env, 0xD18E, 1)
            hp_fraction = current_hp / max(max_hp, 1)
        except:
            hp_fraction = 0.0

        party_size = safe_read_memory(env, 0xD163, 0)
        level = safe_read_memory(env, 0xD18C, 1)

        money = (safe_read_memory(env, 0xD347, 0) * 65536 +
                 safe_read_memory(env, 0xD348, 0) * 256 +
                 safe_read_memory(env, 0xD349, 0))

        battle_mode = safe_read_memory(env, 0xD057, 0)
        in_battle = battle_mode != 0

        return {
            'position': (x_pos, y_pos, map_id),
            'badges': badge_count,
            'hp_fraction': hp_fraction,
            'party_size': party_size,
            'level': level,
            'money': money,
            'in_battle': in_battle,
            'valid': x_pos != -1
        }

    except Exception as e:
        print(f"⚠️ 获取游戏统计信息失败: {e}")
        return {
            'position': (-1, -1, -1),
            'badges': 0,
            'hp_fraction': 0.0,
            'party_size': 0,
            'level': 1,
            'money': 0,
            'in_battle': False,
            'valid': False
        }


def cleanup_memory():
    """手动清理内存"""
    gc.collect()
    if hasattr(gc, 'collect'):
        for i in range(3):  # 多次回收
            gc.collect()


def main():
    print("=" * 60)
    print("🎮 Pokémon 模型观看 (内存优化版)")
    print("=" * 60)

    # ✅ 添加内存监控
    memory_monitor = MemoryMonitor()

    model_path = r'D:\PyCharm 2025.1\code_bailin\PokemonRedExperiments-master\PokemonRedExperiments-master\v5\runs\poke_159744_steps.zip'

    try:
        model = PPO.load(model_path, device="cpu")
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    env_config = {
        'headless': False,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': r'D:\PyCharm 2025.1\code_bailin\PokemonRedExperiments-master\PokemonRedExperiments-master\v5\manual_play_progress_GRASS_BEFORE_BATTLE.state',
        'max_steps': 20000,  # ← 增加最大步数
        'print_rewards': False,
        'save_video': False,
        'fast_video': True,  # ← 启用快速模式
        'session_path': Path("watch_runs"),
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25,
    }

    print("\n🎮 创建环境...")
    try:
        env = make_watch_env(env_config, rank=0)
        print("✅ 环境创建成功")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n🎯 开始观看游戏")
    print("💡 提示：现在有内存监控和异常处理")
    print("=" * 60)

    try:
        obs, info = env.reset(seed=42)
        print("✅ 环境重置完成")

        time.sleep(1)

        step_count = 0
        max_steps = 20000  # ← 增加最大步数
        total_reward = 0.0
        last_cleanup = 0

        while step_count < max_steps:
            try:
                # ✅ 添加详细的异常处理
                try:
                    action, _ = model.predict(obs, deterministic=False)
                except Exception as e:
                    print(f"⚠️ 模型预测失败: {e}")
                    action = env.action_space.sample()  # 随机动作作为备用

                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    print(f"⚠️ 环境步进失败: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                step_count += 1
                total_reward += reward

                # ✅ 内存监控和清理
                if step_count % 100 == 0:
                    memory_info = memory_monitor.get_memory_info()

                    if memory_monitor.should_clean(memory_info['current_mb']):
                        print(f"🔧 内存清理 - 当前: {memory_info['current_mb']:.1f}MB")
                        cleanup_memory()
                        last_cleanup = step_count

                # ✅ 定期输出状态
                if step_count % 50 == 0:
                    memory_info = memory_monitor.get_memory_info()
                    stats = get_detailed_stats(env)

                    if stats['valid']:
                        x, y, map_id = stats['position']
                        print(f"步数: {step_count:5d} | 内存: {memory_info['current_mb']:6.1f}MB | "
                              f"位置: ({x:3d},{y:3d},地图{map_id:2d}) | "
                              f"奖励: {reward:6.2f} (总:{total_reward:7.1f}) | "
                              f"徽章: {stats['badges']} | 等级: {stats['level']:2d} | "
                              f"{'⚔️战斗' if stats['in_battle'] else '🚶探索'}")
                    else:
                        print(f"步数: {step_count:5d} | 内存: {memory_info['current_mb']:6.1f}MB | "
                              f"奖励: {reward:6.2f} | 总奖励: {total_reward:7.1f}")

                time.sleep(0.05)  # ← 稍微减少延迟

                if terminated or truncated:
                    print("✅ 游戏正常结束")
                    final_stats = get_detailed_stats(env)
                    if final_stats['valid']:
                        print(f"🏆 最终状态:")
                        print(f"   徽章数: {final_stats['badges']}")
                        print(f"   等级: {final_stats['level']}")
                        print(f"   金钱: ${final_stats['money']:,}")
                        print(f"   总步数: {step_count}")
                        print(f"   总奖励: {total_reward:.2f}")
                        print(f"   峰值内存: {memory_monitor.peak_memory:.1f}MB")
                    break

            except KeyboardInterrupt:
                print("\n⏸️ 用户中断")
                break
            except Exception as e:
                print(f"⚠️ 步骤执行出错 (步数: {step_count}): {e}")
                import traceback
                traceback.print_exc()

                # 尝试恢复而不是直接退出
                print("🔄 尝试恢复...")
                time.sleep(1)
                continue

        print(f"\n🏁 游戏循环结束 - 总步数: {step_count}")

    except Exception as e:
        print(f"❌ 游戏过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔧 清理资源...")
        try:
            env.close()
        except Exception as e:
            print(f"⚠️ 环境关闭异常: {e}")

        cleanup_memory()
        print("✅ 完成！")


if __name__ == "__main__":
    main()