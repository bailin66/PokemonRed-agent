from pyboy import PyBoy
import time

pyboy = PyBoy(r"D:\PyCharm 2025.1\code_bailin\PokemonRedExperiments-master\PokemonRedExperiments-master\PokemonRed.gb",
              window="SDL2")

with open(r"D:\PyCharm 2025.1\code_bailin\PokemonRedExperiments-master\PokemonRedExperiments-master\v5\manual_play_progress_GRASS_BEFORE_BATTLE.state", "rb") as f:
    pyboy.load_state(f)


class BattleHitDetector:
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.last_damage = 0
        self.last_move_id = 0
        self.attack_processed = False
        # 添加上一次的位置记录
        self.last_position = None

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def is_in_battle(self):
        return self.read_m(0xD057) != 0

    def get_damage_to_deal(self):
        """D0D8 - 即将造成的伤害值"""
        return self.read_m(0xD0D8)

    def check_critical_or_ohko(self):
        """D05E - 暴击/一击必杀标志"""
        return self.read_m(0xD05E)

    def check_move_missed(self):
        """D05F - 未命中标志"""
        return self.read_m(0xD05F) != 0

    def get_player_move_used(self):
        """CCDC - 玩家使用的招式ID"""
        return self.read_m(0xCCDC)

    def get_player_position(self):
        """获取玩家坐标"""
        x = self.read_m(0xD362)
        y = self.read_m(0xD361)
        map_id = self.read_m(0xD35E)
        return x, y, map_id

    def update(self):
        """
        返回: (result, damage, position, moved)
            result: 'hit', 'miss', 'crit', 'ohko', None
            damage: 伤害值（miss时为0）
            position: (x, y, map_id)
            moved: 是否移动了
        """
        x, y, map_id = self.get_player_position()
        current_position = (x, y, map_id)

        # 检查是否移动
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

        # 新招式 = 新攻击
        if current_move != 0 and current_move != self.last_move_id:
            self.last_move_id = current_move
            self.attack_processed = False

        # 同一次攻击只处理一次
        if self.attack_processed or current_move == 0:
            return None, 0, current_position, moved

        # 伤害值从0变成非0 = 命中
        if current_damage > 0 and current_damage != self.last_damage:
            self.last_damage = current_damage
            self.attack_processed = True

            # 检查未命中
            if self.check_move_missed():
                return 'miss', 0, current_position, moved

            # 检查暴击/一击必杀
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

        # 只有移动时才输出坐标
        if moved:
            print(f"Player Position - X: {x}, Y: {y}, Map ID: {map_id}")


except KeyboardInterrupt:
    pass

pyboy.stop()