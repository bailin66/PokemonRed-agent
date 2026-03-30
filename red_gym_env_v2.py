import os

# os.environ['SDL_VIDEODRIVER'] = 'dummy'  # ✅ 添加这两行
# os.environ['SDL_AUDIODRIVER'] = 'dummy'
import uuid
import json
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
# from pyboy.logger import log_level
import mediapy as media
from einops import repeat

import gymnasium as gym
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from global_map import local_to_global, GLOBAL_MAP_SHAPE

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

event_flags_start = 0xD747
event_flags_end = 0xD87E  # expand for SS Anne # old - 0xD7F6
museum_ticket = (0xD754, 0)

# ========== 地址映射常量 ==========
BATTLE_ADDRESSES = {
    # ========== 战斗核心地址 ==========
    'is_in_battle': 0xD057,  # 战斗类型标志
    'damage_to_deal': 0xD0D8,  # ✅ 即将造成的伤害值
    'critical_flag': 0xD05E,  # 暴击/一击必杀标志 (0=普通, 1=暴击, 2=OHKO)
    'miss_flag': 0xD05F,  # 未命中标志 (!=0 = 未命中)
    'player_move_used': 0xCCDC,  # 玩家使用的招式ID

    # ========== 玩家宝可梦（队伍第一只）==========
    'player_current_hp_hi': 0xD16C,
    'player_current_hp_lo': 0xD16D,
    'player_level': 0xD18C,
    'player_max_hp_hi': 0xD18D,
    'player_max_hp_lo': 0xD18E,

    'player_battle_current_hp_hi': 0xD015,  # 战斗中的玩家当前HP高位
    'player_battle_current_hp_lo': 0xD016,  # 战斗中的玩家当前HP低位
    'player_battle_max_hp_hi': 0xD023,      # 战斗中的玩家最大HP高位
    'player_battle_max_hp_lo': 0xD024,      # 战斗中的玩家最大HP低位

    # ========== 队伍信息 ==========
    'party_size': 0xD163,
}


class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = 3
        self.config = config or {}
        self.reward_calculator = self.config.get('reward_calculator')
        self.use_optimized_rewards = self.config.get('use_optimized_rewards', False)
        # ============= 新增：问题10好奇心支持 =============
        self.curiosity = self.config.get('curiosity')
        self.use_curiosity = self.config.get('use_curiosity', False)
        self.prediction_error = 0.5

        # ============= 新增：问题8多任务支持 =============
        self.multi_objectives = self.config.get('multi_objectives', {})
        self.use_multi_task = self.config.get('use_multi_task', False)

        # 初始化多任务计数器
        self.visited_count = 0
        self.battle_wins = 0
        self.total_battles = 0

        # 新增：特征提取
        self.feature_extractor = self.config.get('feature_extractor')
        self.use_feature_extraction = self.config.get('use_feature_extraction', False)

        self.battle_calculator = self.config.get('battle_calculator')
        self.use_battle_optimization = self.config.get('use_battle_optimization', False)

        if self.use_optimized_rewards and self.reward_calculator:
            print("✅ 使用优化的奖励计算")
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        self.essential_map_locations = {
            v: i for i, v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.enc_freqs = 8

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad * 4, self.coords_pad * 4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks)
            }
        )

        head = "null" if config["headless"] else "SDL2"
        print('head:',head)

        # log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            window=head,
        )

        # self.screen = self.pyboy.botsupport_manager().screen()
        self.pyboy.set_emulation_speed(0)
        # if not config["headless"]:
        #     self.pyboy.set_emulation_speed(6)
        # ============= 优化奖励支持（新增） =============
        self.config = config or {}
        self.reward_calculator = self.config.get('reward_calculator')
        self.use_optimized_rewards = self.config.get('use_optimized_rewards', False)
        # ← 新增：探索奖励
        self.exploration_reward = self.config.get('exploration_reward')
        self.use_exploration_optimization = self.config.get('use_exploration_optimization', False)
        self.visited_states = set()  # ← 追踪已访问的状态

        if self.use_optimized_rewards and self.reward_calculator:
            print("✅ 环境已启用优化奖励")
            # ← 新增：打印探索奖励状态
        if self.use_exploration_optimization and self.exploration_reward:
            print("✅ 环境已启用自适应探索奖励")

        # ========== 一号道馆入口位置 ==========
        self.first_gym_entrance_position = {
            2: (16, 18)  # 一号道馆入口位置
        }
        self.min_first_gym_entrance_distance = None
        self.first_gym_entrance_distance = None
        self.last_first_gym_entrance_distance = None
        self.if_has_been_to_first_gym_entrance = False

        # ========== 关键事件奖励字典 ==========
        self.key_events = {
            # ===== 第一道馆：石英市 (Pewter Gym) - Brock =====
            "beat_brock": {
                "bit_address": (0xD755, 7),
                "reward": 150,
                "completed": False,
                "description": "击败Brock(石英市道馆主)"
            },

            "got_tm34": {
                "bit_address": (0xD755, 6),
                "reward": 20,
                "completed": False,
                "description": "从Brock获得TM34"
            },
        }

        self.battle_calculator = self.config.get('battle_calculator')
        self.use_battle_optimization = self.config.get('use_battle_optimization', False)

        # ✅ 战斗追踪变量
        self.in_battle = False
        self.battle_started_player_hp = 0
        self.battle_started_opponent_hp = 0
        self.battle_turn_start = 0
        self.opponent_pokemon_id = 0
        self.opponent_pokemon_level = 0
        self.last_recorded_move = None

        # ============= 捕捉系统初始化 =============
        self.pokemon_catch_count = 0
        self.unique_pokemon_caught = set()
        self.catch_location_map = {}
        self.successful_catch_count = 0

        # 菜单追踪
        self.last_battle_menu_item = 0
        self.last_party_size = 0
        self.last_ball_count = 0

        # 捕捉尝试追踪
        self.catch_attempts = []  # [(timestamp, success), ...]
        self.current_catch_attempt = None

    def reset(self, seed=None, options={}):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats = []

        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)

        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        # experiment!
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1

        self.min_first_gym_entrance_distance = None
        self.first_gym_entrance_distance = None
        self.last_first_gym_entrance_distance = None
        self.if_has_been_to_first_gym_entrance = False

        # ✅ 战斗追踪变量
        self.in_battle = False
        self.battle_started_player_hp = 0
        self.battle_started_opponent_hp = 0
        self.battle_turn_start = 0
        self.opponent_pokemon_id = 0
        self.opponent_pokemon_level = 0
        self.last_recorded_move = None

        self.pokemon_catch_count = 0
        self.unique_pokemon_caught = set()
        self.catch_location_map = {}
        self.successful_catch_count = 0

        # 菜单追踪
        self.last_battle_menu_item = 0
        self.last_party_size = 0
        self.last_ball_count = 0

        # 捕捉尝试追踪
        self.catch_attempts = []  # [(timestamp, success), ...]
        self.current_catch_attempt = None

        return self._get_obs(), {}

    def detect_catch_state_transition(self):
        """✅ 完整的捕捉状态机"""

        # 状态转换：
        # 1. 在野生战斗中 → 2. 选择"道具" → 3. 投掷球 → 4. 队伍大小改变

        if not self.is_in_battle():
            self.current_catch_attempt = None
            return 0

        current_party_size = self.read_m(0xD163)
        battle_menu_item = self.read_m(0xCC26)

        # ========== 状态1：开始捕捉尝试 ==========
        if self.current_catch_attempt is None:
            if battle_menu_item == 1:  # 选择"道具"
                self.current_catch_attempt = {
                    "start_time": self.step_count,
                    "start_party_size": current_party_size,
                    "start_ball_count": self.get_ball_count(),
                    "location": self.get_game_coords(),
                    "success": False
                }

        # ========== 状态2：检查捕捉结果 ==========
        if self.current_catch_attempt is not None:
            # 若队伍大小增加 → 捕捉成功
            if current_party_size > self.current_catch_attempt["start_party_size"]:
                self.current_catch_attempt["success"] = True
                self.current_catch_attempt["end_time"] = self.step_count

                # 记录成功的捕捉
                self.catch_attempts.append(self.current_catch_attempt)
                self.successful_catch_count += 1

                reward = 150.0  # 成功捕捉高奖励

                if self.print_rewards:
                    duration = self.current_catch_attempt["end_time"] - self.current_catch_attempt["start_time"]
                    print(f"🎉 捕捉成功! 耗时: {duration}步 | 奖励: +{reward}")

                self.current_catch_attempt = None
                return reward

            # 若战斗结束且没有捕捉成功 → 捕捉失败
            elif not self.is_in_battle():
                self.current_catch_attempt["success"] = False
                self.catch_attempts.append(self.current_catch_attempt)
                self.current_catch_attempt = None
                return -10.0  # 失败惩罚

        return 0

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        """✅ 优化：只在需要时渲染"""
        # 如果不需要保存视频且是 headless，直接返回空数组
        if self.headless and not self.save_video:
            return np.zeros(self.output_shape[:2] + (1,), dtype=np.uint8)

        # 获取屏幕图像
        screen_pil = self.pyboy.screen_image()
        game_pixels_render = np.array(screen_pil)

        if len(game_pixels_render.shape) == 2:
            game_pixels_render = game_pixels_render[:, :, np.newaxis]

        game_pixels_render = game_pixels_render[:, :, 0:1]

        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2, 2, 1))
            ).astype(np.uint8)

        return game_pixels_render

    def _get_obs(self):
        """✅ 完全不渲染版本（仅用于训练）"""
        # ✅ 跳过渲染，使用空白屏幕
        if self.headless and not self.save_video:
            # 使用全黑屏幕（或者从内存直接读取游戏状态）
            screen = np.zeros(self.output_shape[:2] + (1,), dtype=np.uint8)
        else:
            screen = self.render()

        self.update_recent_screens(screen)

        # 其余代码不变
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions
        }

        return observation

    def detect_menu_open(self):
        """
        改进的菜单检测 - 修复问题

        核心改变：
        1. 追踪菜单状态（打开/关闭）
        2. 持续惩罚菜单打开期间的所有步骤
        3. 增加惩罚力度
        """
        try:
            # 初始化菜单状态追踪变量
            if not hasattr(self, 'menu_open_last_frame'):
                self.menu_open_last_frame = False
            if not hasattr(self, 'menu_open_penalty_count'):
                self.menu_open_penalty_count = 0

            # ========== 1. 战斗中？ ==========
            in_battle = self.read_m(0xD057) != 0
            if in_battle:
                self.menu_open_last_frame = False
                return 0

            # ========== 2. 初始宝可梦选择？ ==========
            map_id = self.read_m(0xD35E)
            party_count = self.read_m(0xD163)
            selecting_starter = (map_id == 40 and party_count == 0)

            if selecting_starter:
                self.menu_open_last_frame = False
                return 0

            # ========== 3. 检测菜单是否打开 ==========
            # 关键：检查多个菜单标志
            menu_open = False

            # 检查0xCC28（主菜单）
            last_menu_item_id = self.read_m(0xCC28)
            if last_menu_item_id > 0:
                menu_open = True

            # 检查0xCC2D（START/战斗菜单光标）- 辅助判断
            menu_cursor = self.read_m(0xCC2D)
            if menu_cursor != 0xFF:  # 0xFF表示菜单不存在
                menu_open = True

            # ========== 4. 持续惩罚打开的菜单 ==========
            penalty = 0

            if menu_open:
                # 菜单打开，持续惩罚
                penalty = -10.0  # 提高惩罚力度
                self.menu_open_penalty_count += 1

            elif self.menu_open_last_frame:
                # 刚关闭菜单，额外惩罚（鼓励快速关闭）
                penalty = -5
                self.menu_open_penalty_count = 0

            # 更新菜单状态
            self.menu_open_last_frame = menu_open

            return penalty

        except Exception as e:
            print(f"菜单检测错误: {e}")
            return 0

    def detect_battle_state(self):
        """✅ 完整的战斗状态检测和奖励计算"""

        # 初始化追踪变量
        if not hasattr(self, 'last_damage_value'):
            self.last_damage_value = 0
            self.last_move_id = 0
            self.attack_processed = False

        battle_reward = 0.0

        # ========== 1. 检测战斗状态 ==========
        current_in_battle = self.is_in_battle()

        # 战斗开始
        if current_in_battle and not self.in_battle:
            self.in_battle = True
            self.battle_started_player_hp = self.get_player_battle_hp()
            self.attack_processed = False
            if self.print_rewards:
                print(f"\n⚔️ 战斗开始！初始HP: {self.battle_started_player_hp}")

        # 战斗结束
        elif not current_in_battle and self.in_battle:
            final_hp = self.get_player_battle_hp()

            # 判断胜负
            if final_hp > 0:
                battle_reward += 15.0  # 胜利奖励
                if self.print_rewards:
                    print(f"✅ 战斗胜利！剩余HP: {final_hp}")
            else:
                battle_reward -= 3.0  # 失败惩罚
                if self.print_rewards:
                    print(f"❌ 战斗失败")

            # 重置状态
            self.in_battle = False
            self.last_damage_value = 0
            self.last_move_id = 0
            self.attack_processed = False

        # ========== 2. 战斗中：处理伤害和招式记录 ==========
        if self.in_battle:
            current_damage = self.get_damage_to_deal()
            current_move = self.get_player_move_used()

            # 伤害结算逻辑：
            # - 伤害值 > 0
            # - 伤害值发生变化（新的攻击）
            # - 还没处理过这次攻击
            if (current_damage > 0 and
                    current_damage != self.last_damage_value and
                    not self.attack_processed):

                # 记录伤害
                self.last_damage_value = current_damage
                self.last_move_id = current_move
                self.attack_processed = True

                # ========== 2.1 基础伤害奖励 ==========
                # 伤害越高，奖励越大（归一化到0-1）
                damage_ratio = min(current_damage / 100.0, 1.0)
                base_reward = damage_ratio * 10.0
                battle_reward += base_reward

                # ========== 2.2 高伤害额外奖励 ==========
                if current_damage > 10:
                    bonus = (current_damage - 10)
                    battle_reward += bonus

                if self.print_rewards:
                    print(f"💥 造成伤害: {current_damage} | 奖励: +{base_reward:.2f}")

                # ========== 2.3 记录到战斗学习系统 ==========
                if self.use_battle_optimization and self.battle_calculator:
                    try:
                        if current_move > 0:
                            move_name = self.get_move_name(current_move)

                            # ✅ 只传招式名和伤害值
                            self.battle_calculator.action_evaluator.record_move_damage(
                                move_name=move_name,
                                damage_dealt=current_damage
                            )

                            if self.print_rewards:
                                avg_dmg = self.battle_calculator.action_evaluator.get_move_average_damage(move_name)
                                print(f"📊 {move_name} 平均伤害: {avg_dmg:.1f}")

                    except Exception as e:
                        if self.print_rewards:
                            print(f"⚠️ 战斗记录错误: {e}")

            # 伤害值归零 = 新回合开始，重置处理标志
            elif current_damage == 0 and self.last_damage_value > 0:
                self.attack_processed = False

        return battle_reward

    def get_move_name(self, move_id):
        """✅ 完整的招式名称查询表"""
        MOVE_NAMES = {
            0: "NONE",
            1: "POUND",
            2: "KARATE_CHOP",
            3: "DOUBLE_SLAP",
            4: "COMET_PUNCH",
            5: "MEGA_PUNCH",
            6: "PAY_DAY",
            7: "FIRE_PUNCH",
            8: "ICE_PUNCH",
            9: "THUNDER_PUNCH",
            10: "SCRATCH",
            11: "VICE_GRIP",
            12: "GUILLOTINE",
            13: "RAZOR_WIND",
            14: "SWORDS_DANCE",
            15: "CUT",
            16: "GUST",
            17: "WING_ATTACK",
            18: "WHIRLWIND",
            19: "FLY",
            20: "BIND",
            21: "SLAM",
            22: "VINE_WHIP",  # ✅ 妙蛙种子
            23: "STOMP",
            24: "DOUBLE_KICK",
            25: "MEGA_KICK",
            26: "JUMP_KICK",
            27: "KICK",
            28: "MEGA_DRAIN",
            29: "LEECH_SEED",
            30: "GROWTH",
            31: "RAZOR_LEAF",
            32: "SOLAR_BEAM",
            33: "POISONPOWDER",
            34: "STUN_SPORE",
            35: "SLEEP_POWDER",
            36: "PETAL_DANCE",
            37: "STRING_SHOT",
            38: "DRAGON_RAGE",
            39: "FIRE_SPIN",
            40: "THUNDERSHOCK",
            41: "THUNDERBOLT",
            42: "THUNDER_WAVE",
            43: "THUNDER",
            44: "ROCK_THROW",
            45: "EARTHQUAKE",
            46: "FISSURE",
            47: "DIG",
            48: "TOXIC",
            49: "CONFUSION",
            50: "PSYCHIC",
            51: "HYPNOSIS",
            52: "MEDITATE",
            53: "AGILITY",
            54: "QUICK_ATTACK",
            55: "RAGE",
            56: "TACKLE",  # ✅ 基础招式
            57: "BODY_SLAM",
            58: "WRAP",
            59: "TAKE_DOWN",
            60: "THRASH",
            61: "DOUBLE_EDGE",
            62: "BUBBLEBEAM",
            63: "WATER_GUN",
            64: "HYDRO_PUMP",
            65: "SURF",
            66: "ICE_BEAM",
            67: "BLIZZARD",
            68: "PSYBEAM",
            69: "AURORA_BEAM",
            70: "HYPER_BEAM",
            71: "PECK",
            72: "DRILL_PECK",
            73: "SUBMISSION",
            74: "LOW_KICK",
            75: "COUNTER",
            76: "SEISMIC_TOSS",
            77: "STRENGTH",
            78: "ABSORB",
            79: "ACID",
            80: "EMBER",
            81: "FLAMETHROWER",
            82: "MIST",
            83: "WATER_SPORT",
            84: "ICE_SHARD",
            85: "SMOKESCREEN",
            86: "MACHOP",
            87: "FURY_CUTTER",
            88: "ACID_SPRAY",
            89: "SLUDGE_BOMB",
            90: "BONE_CLUB",
            91: "BONE_RUSH",
            92: "FURY_ATTACK",
            93: "HORN_DRILL",
            94: "TUSK",
            95: "HORN_ATTACK",
            96: "HEADBUTT",
            97: "TAIL_WHIP",
            98: "LEER",
            99: "BITE",
            100: "GROWL",
            101: "ROAR",
            102: "SING",
            103: "SUPERSONIC",
            104: "SONIC_BOOM",
            105: "DISABLE",
            106: "ACID_ARMOR",
            107: "EMOTE",
            108: "FLASH",
            109: "PROTECT",
            110: "AMNESIA",
            111: "MINIMIZE",
            112: "MIRROR_COAT",
            113: "SELF_DESTRUCT",
            114: "EXPLOSION",
            115: "SPORE",
            116: "BARRAGE",
            117: "LEECH_LIFE",
            118: "LOVELY_KISS",
            119: "SKY_ATTACK",
            120: "TRANSFORM",
        }

        if move_id in MOVE_NAMES:
            return MOVE_NAMES[move_id]
        else:
            return f"MOVE_{move_id}"

    def get_battle_turn_count(self):
        """✅ 获取当前战斗回合数"""
        # 从内存中读取回合计数（如果有的话）
        # 或者通过记录的伤害历史推算
        if self.use_battle_optimization and self.battle_calculator:
            move_stats = self.battle_calculator.action_evaluator.move_usage_count
            return sum(move_stats.values())
        return 0

    def get_first_gym_entrance_approach_reward(self):
        """接近 first_gym_entrance 位置的渐进式奖励"""

        if self.if_has_been_to_first_gym_entrance:  # 已经到过gym_entrance，停止奖励
            return 0

        x, y, map_n = self.get_game_coords()
        if map_n not in self.first_gym_entrance_position:   # 不在小镇，停止奖励
            return 0

        # 已经打败过blue，并且还没去过北出口
        target_x, target_y = self.first_gym_entrance_position[map_n]
        current_distance = abs(x - target_x) + abs(y - target_y)

        reward = 0

        if self.min_first_gym_entrance_distance is None or current_distance < self.min_first_gym_entrance_distance:
            if self.min_first_gym_entrance_distance is not None:
                improvement = self.min_first_gym_entrance_distance - current_distance
                reward = improvement * 10.0

            self.min_first_gym_entrance_distance = current_distance

            if current_distance == 0:
                reward += 15.0
                self.if_has_been_to_first_gym_entrance = True

        elif self.last_first_gym_entrance_distance is not None and current_distance > self.last_first_gym_entrance_distance:
            reward = -2.0

        self.last_first_gym_entrance_distance = current_distance

        return reward

    def step(self, action):

        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()

        self.party_size = self.read_m(0xD163)

        reward = self.update_reward()

        battle_reward = self.detect_battle_state()
        reward += battle_reward

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_and_print_info(step_limit_reached, obs)

        # create a map of all event flags set, with names where possible
        # if step_limit_reached:
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        # TODO this currently seems to be broken!
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1

        # ============= 优化奖励处理（新增） =============
        if self.use_optimized_rewards and self.reward_calculator:
            try:
                reward_dict = {
                    "exploration": 0.0,
                    "battle": reward if hasattr(self, 'in_battle') and self.in_battle else 0.0,
                    "progress": reward if not (hasattr(self, 'in_battle') and self.in_battle) else 0.0,
                }
                optimized_reward, _ = self.reward_calculator.calculate_total_reward(reward_dict)
                reward = optimized_reward
            except Exception as e:
                pass
        # ============= 特征提取（新增） =============
        if self.use_feature_extraction and self.feature_extractor:
            try:
                features = self.feature_extractor.extract(obs)
                # 可以在这里使用提取的特征
            except Exception as e:
                pass

        # ============= 新增：探索奖励处理 =============
        if self.use_exploration_optimization and self.exploration_reward:
            try:
                # 将当前状态哈希化并添加到访问集合
                # 使用obs的前100个字符作为状态哈希
                state_hash = hash(str(obs)[:100])
                self.visited_states.add(state_hash)

                # 计算探索奖励
                exploration_bonus = self.exploration_reward.calculate(self.visited_states)

                # 将探索奖励加到总奖励上（权重为0.1）
                reward += exploration_bonus * 0.1

            except Exception as e:
                # 如果出错，继续使用原始奖励
                pass

        # ============= 战斗优化（新增） =============
        if self.use_battle_optimization and self.battle_calculator:
            try:
                # 检测是否在战斗中
                if hasattr(self, 'in_battle') and self.in_battle:
                    battle_state = {
                        "player_hp": getattr(self, 'player_hp', 0),
                        "opponent_hp": getattr(self, 'opponent_hp', 0),
                    }
                    battle_reward = self.battle_calculator.calculate(battle_state)
                    reward = max(reward, battle_reward)  # 使用较大的奖励
            except Exception as e:
                pass
        # ============= 新增：问题8多任务奖励计算 =============
        if self.use_multi_task and self.multi_objectives:
            try:
                multi_task_reward = 0.0

                # 目标1：探索效率
                exploration_weight = self.multi_objectives.get("exploration_efficiency", {}).get("weight", 0.3)
                if hasattr(self, 'visited_states'):
                    exploration_bonus = len(self.visited_states) * 0.01 * exploration_weight
                    multi_task_reward += exploration_bonus

                # 目标2：战斗性能
                battle_weight = self.multi_objectives.get("battle_performance", {}).get("weight", 0.4)
                if hasattr(self, 'in_battle') and self.in_battle and reward > 0:
                    self.battle_wins += 1
                if hasattr(self, 'in_battle') and self.in_battle:
                    self.total_battles += 1
                    win_rate = self.battle_wins / max(self.total_battles, 1)
                    battle_reward = win_rate * battle_weight
                    multi_task_reward += battle_reward

                # 目标3：地图覆盖
                coverage_weight = self.multi_objectives.get("map_coverage", {}).get("weight", 0.3)
                self.visited_count += 1
                coverage_bonus = min(self.visited_count / 10000, 1.0) * coverage_weight
                multi_task_reward += coverage_bonus

                reward += multi_task_reward * 0.1

            except Exception as e:
                pass

        # ============= 新增：问题10好奇心奖励 =============
        if self.use_curiosity and self.curiosity:
            try:
                # 简单的预测误差计算（基于观测变化）
                state_hash = hash(str(obs)[:100])

                # 估计预测误差（可以改进）
                self.prediction_error = abs(reward) * 0.1 + 0.5

                # 更新好奇心模块
                self.curiosity.update_prediction(state_hash, self.prediction_error)

                # 获取好奇心奖励
                curiosity_bonus = self.curiosity.get_curiosity_reward(state_hash)

                # 添加到总奖励
                reward += curiosity_bonus * 0.05

            except Exception as e:
                pass

        return obs, reward, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        """✅ PyBoy 2.4.0 兼容版本"""
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])

        press_step = 8
        # disable rendering when we don't need it
        render_screen = self.save_video or not self.headless

        # ✅ 修改：PyBoy 2.4.0 的 tick() 无参数，每次调用一步
        # 按下按钮的步数
        for _ in range(press_step):
            self.pyboy.tick()

        # 松开按钮
        self.pyboy.send_input(self.release_actions[action])

        # 剩余的步数
        for _ in range(self.act_freq - press_step - 1):
            self.pyboy.tick()

        # 最后一步
        self.pyboy.tick()

        # 如果需要保存视频
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
            }
        )

    def start_video(self):

        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:, :, 0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:, :, 0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )

    def get_game_coords(self):
        """✅ 使用 read_m() 方法获取坐标"""
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        # if not in battle
        if self.read_m(0xD057) == 0:
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1
            # self.seen_coords[coord_string] = self.step_count

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        else:
            out = self.explore_map[
                  c[0] - self.coords_pad:c[0] + self.coords_pad,
                  c[1] - self.coords_pad:c[1] + self.coords_pad
                  ]
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)

    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:, :, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:, :, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:, :, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

    def read_m(self, addr):
        """使用 PyBoy 2.4.0 的 get_memory_value() 方法"""
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 4,
            # "level": self.reward_scale * self.get_levels_reward(),
            "heal": self.reward_scale * self.total_healing_rew * 10,
            # "op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            # "dead": self.reward_scale * self.died_count * -0.1,
            "badge": self.reward_scale * self.get_badges() * 10,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1,

            "first_gym_entrance_approach": self.reward_scale * self.get_first_gym_entrance_approach_reward(),
            "menu_penalty": self.detect_menu_open(),  # 惩罚非战斗菜单

            "catch": self.reward_scale * self.get_catch_reward(),
            "catch_diversity": self.reward_scale * self.get_catch_diversity_reward(),

            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05
        }

        return state_scores

    def get_catch_reward(self):
        """捕捉宝可梦的即时奖励（只奖励新捕获）"""
        current_count = len(self.unique_pokemon_caught)

        # 初始化
        if not hasattr(self, 'last_caught_count'):
            self.last_caught_count = current_count
            return 0

        # 计算增量
        new_catches = current_count - self.last_caught_count

        if new_catches > 0:
            # 只奖励新捕获的
            reward = new_catches * 50.0
            self.last_caught_count = current_count
            return reward
        return 0

    def get_catch_diversity_reward(self):
        """捕捉多样性里程碑"""
        unique_count = len(self.unique_pokemon_caught)

        if not hasattr(self, 'last_catch_milestone'):
            self.last_catch_milestone = 0

        milestones = [1, 3, 5, 10, 15, 20]

        for milestone in milestones:
            if self.last_catch_milestone < milestone <= unique_count:
                bonus = milestone * 15
                self.last_catch_milestone = unique_count
                return bonus
        return 0

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
                max([
                    self.read_m(a)
                    for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
                ])
                - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_damage_to_deal(self):
        """从内存读取即将造成的伤害"""
        return self.read_m(BATTLE_ADDRESSES['damage_to_deal'])

    def get_player_move_used(self):
        """从内存读取玩家使用的招式ID"""
        return self.read_m(BATTLE_ADDRESSES['player_move_used'])

    def get_player_battle_hp(self):
        """获取战斗中的玩家HP"""
        hi = self.read_m(BATTLE_ADDRESSES['player_battle_current_hp_hi'])
        lo = self.read_m(BATTLE_ADDRESSES['player_battle_current_hp_lo'])
        return (hi << 8) | lo

    def get_player_max_hp(self):
        """获取玩家最大HP"""
        hi = self.read_m(BATTLE_ADDRESSES['player_battle_max_hp_hi'])
        lo = self.read_m(BATTLE_ADDRESSES['player_battle_max_hp_lo'])
        return (hi << 8) | lo

    def is_in_battle(self):
        """检测是否在战斗中"""
        return self.read_m(BATTLE_ADDRESSES['is_in_battle']) != 0

