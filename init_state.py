from pyboy import PyBoy


# ============= 配置 =============
ROM_PATH = r"D:\PyCharm 2025.1\code_bailin\PokemonRedExperiments-master\PokemonRedExperiments-master\PokemonRed.gb"
INIT_STATE = r"D:\PyCharm 2025.1\code_bailin\PokemonRedExperiments-master\PokemonRedExperiments-master\v5\manual_play_progress_GRASS_CATCH_POKEMON.state"
SAVE_STATE = "../manual_play_progress.state"

pyboy = PyBoy(ROM_PATH, window="SDL2")

# 加载初始状态
with open(INIT_STATE, "rb") as f:
    pyboy.load_state(f)


class BattleHitDetector:
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.last_damage = 0
        self.last_move_id = 0
        self.attack_processed = False
        self.last_position = None

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def is_in_battle(self):
        return self.read_m(0xD057) != 0

    def get_damage_to_deal(self):
        return self.read_m(0xD0D8)

    def check_critical_or_ohko(self):
        return self.read_m(0xD05E)

    def check_move_missed(self):
        return self.read_m(0xD05F) != 0

    def get_player_move_used(self):
        return self.read_m(0xCCDC)

    def get_player_position(self):
        x = self.read_m(0xD362)
        y = self.read_m(0xD361)
        map_id = self.read_m(0xD35E)
        return x, y, map_id

    def update(self):
        x, y, map_id = self.get_player_position()
        current_position = (x, y, map_id)

        moved = False
        if self.last_position is None:
            self.last_position = current_position
        elif current_position != self.last_position:
            moved = True
            self.last_position = current_position

        if not self.is_in_battle():
            self.last_damage = 0
            self.last_move_id = 0
            self.attack_processed = False
            return None, 0, current_position, moved

        current_move = self.get_player_move_used()
        current_damage = self.get_damage_to_deal()

        if current_move != 0 and current_move != self.last_move_id:
            self.last_move_id = current_move
            self.attack_processed = False

        if self.attack_processed or current_move == 0:
            return None, 0, current_position, moved

        if current_damage > 0 and current_damage != self.last_damage:
            self.last_damage = current_damage
            self.attack_processed = True

            if self.check_move_missed():
                return 'miss', 0, current_position, moved

            crit = self.check_critical_or_ohko()
            if crit == 1:
                return 'crit', current_damage, current_position, moved
            elif crit == 2:
                return 'ohko', current_damage, current_position, moved
            else:
                return 'hit', current_damage, current_position, moved

        return None, 0, current_position, moved

detector = BattleHitDetector(pyboy)

try:
    while not pyboy.tick():
        result, damage, (x, y, map_id), moved = detector.update()
        print(f"位置: ({x:3d},{y:3d},地图{map_id:2d}) | ")

except KeyboardInterrupt:
    print("\n")
    print("=" * 60)
    print("💾 正在保存游戏进度...")

    try:
        with open(SAVE_STATE, "wb") as f:
            pyboy.save_state(f)
        print(f"✅ 进度已保存到: {SAVE_STATE}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        print("=" * 60)

finally:
    print("👋 游戏结束")
    pyboy.stop()