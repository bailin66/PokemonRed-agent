# timeout_wrapper.py

import gym
import threading
import sys


class TimeoutWrapper(gym.Wrapper):
    """为环境的 step() 和 reset() 添加超时保护"""

    def __init__(self, env, timeout=10):
        super().__init__(env)
        self.timeout = timeout

    def step(self, action):
        """带超时的 step()"""
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = self.env.step(action)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(self.timeout)

        if thread.is_alive():
            print(f"⏰ env.step() 超时（{self.timeout}秒）")
            obs = self.observation_space.sample()
            return obs, -1.0, True, {'timeout': True}

        if exception[0]:
            raise exception[0]

        return result[0]

    def reset(self, **kwargs):
        """带超时的 reset()"""
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = self.env.reset(**kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(self.timeout)

        if thread.is_alive():
            print(f"⏰ env.reset() 超时（{self.timeout}秒）")
            return self.observation_space.sample()

        if exception[0]:
            raise exception[0]

        return result[0]