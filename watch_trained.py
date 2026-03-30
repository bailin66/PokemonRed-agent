import time
from pathlib import Path

from stable_baselines3 import PPO
from red_gym_env_v2 import RedGymEnv


# ========= 你需要改的两项 =========
MODEL_PATH = Path("runs/poke_4079616_steps_zhangbo.zip")
INIT_STATE = Path("../init.state")  # 必须是“从头开始”的 savestate
# =================================


def main():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    assert INIT_STATE.exists(), f"Init state not found: {INIT_STATE}"

    ep_length = 4096

    env_config = {
        "headless": False,              # 打开窗口
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../init.state",  # 关键：从这个存档开始
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,            # 若你的 env 内置录制逻辑，可改 True
        "fast_video": True,
        "session_path": Path("runs"),
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    env = RedGymEnv(env_config)

    # 加载模型（和 SB3 文档一致：PPO.load(path, env=env)）[web:255]
    model = PPO.load(str(MODEL_PATH), env=env, device="cpu")  # 没 GPU 改成 "cpu"

    episodes = 5
    deterministic = True
    base_seed = 123  # 固定种子：方便你复现实验；不影响“从 init_state 开始”

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=base_seed + ep)  # 每局都强制从 init_state reset
            ep_return = 0.0
            ep_steps = 0

            while True:
                # predict(deterministic=True) 用于评估时“少随机性”回放 [web:255]
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_return += float(reward)
                ep_steps += 1

                # 可选：如果画面太快看不清，打开这行（按需调大/调小）
                #time.sleep(0.25)

                if terminated or truncated or ep_steps >= ep_length:
                    print(f"[Episode {ep+1}/{episodes}] steps={ep_steps}, return={ep_return:.3f}")
                    break

    except KeyboardInterrupt:
        print("Stopped by user (Ctrl+C).")

    finally:
        env.close()


if __name__ == "__main__":
    main()
