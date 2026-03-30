"""
优化方案5：战斗微策略学习系统（基于伤害的自适应奖励）
原问题：战斗系统的奖励不适配，缺乏中间奖励信号

改进方案：
1. 独立的战斗策略学习器（微观目标）
2. 分层奖励信号（宏观+微观）
3. 战斗状态检测和上下文感知
4. ✅ 基于实际伤害的自适应学习（无需类型信息）

核心思想：
- AI通过观察伤害大小自动学习招式效果
- 高伤害招式 → 高奖励 → 优先使用
- 无需硬编码属性克制关系
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
from enum import Enum


class BattlePhase(Enum):
    """战斗阶段枚举"""
    PRE_BATTLE = 0
    BATTLE_START = 1
    PLAYER_TURN = 2
    OPPONENT_TURN = 3
    BATTLE_END = 4
    VICTORY = 5
    DEFEAT = 6


class BattleState:
    """战斗状态"""

    def __init__(
        self,
        phase: BattlePhase,
        player_hp: int,
        enemy_hp: int,
        player_level: int,
        enemy_level: int,
        player_pokemon_id: int,
        enemy_pokemon_id: int,
        player_status: str = "normal",
        enemy_status: str = "normal",
    ):
        self.phase = phase
        self.player_hp = player_hp
        self.enemy_hp = enemy_hp
        self.player_level = player_level
        self.enemy_level = enemy_level
        self.player_pokemon_id = player_pokemon_id
        self.enemy_pokemon_id = enemy_pokemon_id
        self.player_status = player_status
        self.enemy_status = enemy_status
        self.timestamp = 0

    def get_state_vector(self) -> np.ndarray:
        """获取状态向量（用于神经网络输入）"""
        return np.array([
            self.phase.value,
            self.player_hp / 255.0,
            self.enemy_hp / 255.0,
            self.player_level / 100.0,
            self.enemy_level / 100.0,
            (self.player_hp > 0),
            (self.enemy_hp > 0),
        ], dtype=np.float32)


class BattleActionEvaluator:
    """✅ 纯基于伤害的评估器"""

    def __init__(self):
        self.move_damage_history = {}  # {move_name: [damage1, damage2, ...]}
        self.move_usage_count = {}

    def record_move_damage(self, move_name: str, damage_dealt: int):
        """✅ 只需要招式名和伤害"""
        if move_name not in self.move_damage_history:
            self.move_damage_history[move_name] = []
            self.move_usage_count[move_name] = 0

        self.move_damage_history[move_name].append(damage_dealt)
        self.move_usage_count[move_name] += 1

        # 只保留最近20次
        if len(self.move_damage_history[move_name]) > 20:
            self.move_damage_history[move_name].pop(0)

    def get_move_average_damage(self, move_name: str) -> float:
        """获取招式平均伤害"""
        history = self.move_damage_history[move_name]
        return np.mean(history)

    def get_best_move(self, available_moves: List[str]) -> str:
        """返回历史最高伤害的招式"""
        best_move = available_moves[0]
        best_damage = 0

        for move in available_moves:
            avg_damage = self.get_move_average_damage(move)
            if avg_damage > best_damage:
                best_damage = avg_damage
                best_move = move

        return best_move

    def get_move_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {}
        for move_name, damages in self.move_damage_history.items():
            if damages:
                stats[move_name] = {
                    "avg_damage": np.mean(damages),
                    "max_damage": np.max(damages),
                    "min_damage": np.min(damages),
                    "usage_count": self.move_usage_count[move_name],
                }
        return stats

class BattleRewardCalculator:
    """战斗奖励计算器"""

    def __init__(self):
        self.action_evaluator = BattleActionEvaluator()
        self.battle_history = deque(maxlen=100)
        self.battle_stats = {
            "total_battles": 0,
            "won_battles": 0,
            "lost_battles": 0,
            "total_damage_dealt": 0,
            "total_damage_taken": 0,
            "total_turns": 0,
        }

    def calculate_step_reward(
        self,
        current_state: BattleState,
        previous_state: BattleState,
        player_action: str,
        optimal_action: str,
        optimal_score: float
    ) -> Dict[str, float]:
        """✅ 计算步骤奖励（核心：基于实际伤害）"""

        rewards = {
            "action_quality": 0.0,      # 动作选择质量
            "damage_dealt": 0.0,        # 造成伤害奖励
            "damage_efficiency": 0.0,   # 高效伤害额外奖励
            "damage_taken": 0.0,        # 受到伤害惩罚
            "survival": 0.0,            # 生存奖励
            "progress": 0.0,            # 战斗进度奖励
            "exploration": 0.0,         # 探索奖励
        }

        # 1. 动作质量奖励
        if player_action == optimal_action:
            rewards["action_quality"] = 0.1

        # 2. ✅ 伤害奖励（核心机制）
        damage_dealt = previous_state.enemy_hp - current_state.enemy_hp

        if damage_dealt > 0:
            # 记录到历史（用于未来决策）
            self.action_evaluator.record_move_damage(
                player_action,
                current_state.enemy_pokemon_id)

            # 基础伤害奖励（线性）
            damage_ratio = damage_dealt / 255.0
            rewards["damage_dealt"] = damage_ratio * 2.0

            # 高效伤害额外奖励（非线性）,鼓励使用高伤害招式
            if damage_ratio > 0.2:  # 超过20% HP
                efficiency_bonus = (damage_ratio - 0.2) * 5.0
                rewards["damage_efficiency"] = efficiency_bonus

            # 统计
            self.battle_stats["total_damage_dealt"] += damage_dealt

        elif damage_dealt < 0:
            # 敌人回血（罕见情况）
            rewards["damage_dealt"] = damage_dealt / 255.0 * 0.3

        # 3. 受伤惩罚
        damage_taken = previous_state.player_hp - current_state.player_hp

        if damage_taken > 0:
            damage_ratio = damage_taken / 255.0
            rewards["damage_taken"] = -damage_ratio * 0.2

            self.battle_stats["total_damage_taken"] += damage_taken

        # 4. 生存奖励
        if current_state.player_hp > 0:
            hp_ratio = current_state.player_hp / 255.0
            rewards["survival"] = 0.02 * hp_ratio
        else:
            # 死亡严重惩罚
            rewards["survival"] = -2.0

        # 5. 战斗进度奖励
        enemy_hp_decrease = (previous_state.enemy_hp - current_state.enemy_hp) / 255.0
        if enemy_hp_decrease > 0:
            rewards["progress"] = enemy_hp_decrease * 0.3

        # 6. 探索奖励（鼓励尝试不同招式）
        confidence = self.action_evaluator.get_move_confidence(player_action)
        if confidence < 0.3:
            rewards["exploration"] = 0.05

        return rewards

    def calculate_battle_terminal_reward(
        self,
        final_state: BattleState,
        total_turns: int
    ) -> Dict[str, float]:
        """计算终局奖励"""

        rewards = {
            "victory": 0.0,
            "defeat": 0.0,
            "efficiency": 0.0,
            "survival_bonus": 0.0,
        }

        if final_state.phase == BattlePhase.VICTORY:
            # 胜利基础奖励
            rewards["victory"] = 3.0

            # 快速胜利奖励（回合数越少越好）
            if total_turns <= 5:
                efficiency_bonus = (6 - total_turns) * 0.5
                rewards["efficiency"] = efficiency_bonus

            # 高血量胜利奖励
            hp_ratio = final_state.player_hp / 255.0
            if hp_ratio > 0.5:
                rewards["survival_bonus"] = hp_ratio * 0.5

            # 统计
            self.battle_stats["won_battles"] += 1

        elif final_state.phase == BattlePhase.DEFEAT:
            # 失败惩罚
            rewards["defeat"] = -2.0

            # 但如果造成了显著伤害，减少惩罚
            enemy_hp_ratio = final_state.enemy_hp / 255.0
            if enemy_hp_ratio < 0.3:
                rewards["defeat"] += 0.5  # 几乎打赢了

            # 统计
            self.battle_stats["lost_battles"] += 1

        # 统计
        self.battle_stats["total_battles"] += 1
        self.battle_stats["total_turns"] += total_turns

        return rewards

    def get_battle_statistics(self) -> Dict:
        """获取战斗统计"""

        total_battles = max(self.battle_stats["total_battles"], 1)
        won_battles = self.battle_stats["won_battles"]

        stats = {
            **self.battle_stats,
            "win_rate": won_battles / total_battles,
            "avg_turns_per_battle": self.battle_stats["total_turns"] / total_battles,
            "avg_damage_per_battle": self.battle_stats["total_damage_dealt"] / total_battles,
            "damage_efficiency": (
                self.battle_stats["total_damage_dealt"] /
                max(self.battle_stats["total_damage_taken"], 1)
            ),
        }

        # 添加招式统计
        stats["move_statistics"] = self.action_evaluator.get_move_statistics()

        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_battle_statistics()

        print("\n" + "="*60)
        print("📊 战斗统计")
        print("="*60)
        print(f"总战斗数: {stats['total_battles']}")
        print(f"胜利数: {stats['won_battles']}")
        print(f"失败数: {stats['lost_battles']}")
        print(f"胜率: {stats['win_rate']:.1%}")
        print(f"平均回合数: {stats['avg_turns_per_battle']:.1f}")
        print(f"平均伤害: {stats['avg_damage_per_battle']:.1f}")
        print(f"伤害效率: {stats['damage_efficiency']:.2f}x")

        print("\n" + "="*60)
        print("⚔️  招式效果统计")
        print("="*60)

        move_stats = stats.get("move_statistics", {})
        if move_stats:
            # 按平均伤害排序
            sorted_moves = sorted(
                move_stats.items(),
                key=lambda x: x[1]["avg_damage"],
                reverse=True
            )

            for move_name, data in sorted_moves:
                print(f"\n{move_name}:")
                print(f"  平均伤害: {data['avg_damage']:.1%} HP")
                print(f"  伤害范围: {data['min_damage']:.1%} - {data['max_damage']:.1%}")
                print(f"  使用次数: {data['usage_count']}")
                print(f"  置信度: {data['confidence']:.1%}")
        else:
            print("（暂无数据）")


class HierarchicalBattlePolicy:
    """分层战斗策略"""

    def __init__(self):
        # 高层策略：选择使用哪个Pokemon
        self.pokemon_selection_policy = {}

        # 中层策略：战斗中的招式选择
        self.move_selection_policy = {}

        # 低层评估：动作质量评估
        self.action_evaluator = BattleActionEvaluator()

        # 学习历史
        self.policy_update_history = deque(maxlen=100)

    def select_pokemon(
        self,
        player_team: List[int],
        opponent_team: List[int],
        battle_context: Dict
    ) -> int:
        """选择出战的宝可梦"""
        # 简化策略：选择第一个存活的Pokemon
        for pokemon_id in player_team:
            if pokemon_id > 0:
                return pokemon_id

        return player_team[0]

    def select_move(
        self,
        battle_state: BattleState,
        available_moves: List[str]
    ) -> str:
        """选择战斗招式"""
        best_move, _ = self.action_evaluator.get_optimal_action(
            battle_state,
            available_moves
        )

        return best_move

    def update_policy(
        self,
        transition: Dict,
        reward: float
    ):
        """更新策略（用于未来的策略梯度学习）"""
        self.policy_update_history.append({
            "transition": transition,
            "reward": reward,
            "timestamp": len(self.policy_update_history),
        })


class BattleContextManager:
    """战斗上下文管理器"""

    def __init__(self):
        self.current_battle = None
        self.battle_history = deque(maxlen=50)
        self.in_battle = False

    def start_battle(
        self,
        player_pokemon: int,
        enemy_pokemon: int,
        enemy_level: int
    ):
        """开始战斗"""
        self.current_battle = {
            "player_pokemon": player_pokemon,
            "enemy_pokemon": enemy_pokemon,
            "enemy_level": enemy_level,
            "turns": 0,
            "player_damage_dealt": 0,
            "enemy_damage_dealt": 0,
            "start_time": 0,
        }
        self.in_battle = True

    def end_battle(self, victory: bool):
        """结束战斗"""
        if self.current_battle is not None:
            self.current_battle["victory"] = victory
            self.battle_history.append(self.current_battle.copy())
        self.in_battle = False
        self.current_battle = None

    def get_battle_statistics(self) -> Dict:
        """获取战斗管理统计"""
        if len(self.battle_history) == 0:
            return {}

        victories = sum(1 for b in self.battle_history if b.get("victory", False))
        total_battles = len(self.battle_history)

        return {
            "total_battles": total_battles,
            "victories": victories,
            "defeats": total_battles - victories,
            "win_rate": victories / max(total_battles, 1),
            "average_enemy_level": np.mean([b["enemy_level"] for b in self.battle_history]),
        }


# ============= 测试示例 =============

if __name__ == "__main__":
    print("="*60)
    print("✅ 基于伤害的自适应战斗奖励系统")
    print("="*60)
    print("\n核心机制：")
    print("  1. AI观察每个招式造成的实际伤害")
    print("  2. 高伤害招式 → 高奖励 → 优先使用")
    print("  3. 无需属性克制知识，AI自己学习！")
    print("="*60)

    # 初始化组件
    reward_calc = BattleRewardCalculator()
    action_evaluator = reward_calc.action_evaluator
    battle_manager = BattleContextManager()

    # 模拟多场战斗，让AI学习招式效果
    print("\n📚 学习阶段：AI尝试不同招式并学习效果...")

    num_battles = 5

    for battle_num in range(num_battles):
        print(f"\n{'='*60}")
        print(f"战斗 {battle_num + 1}/{num_battles}")
        print(f"{'='*60}")

        battle_manager.start_battle(
            player_pokemon=1,   # 妙蛙种子
            enemy_pokemon=74,   # 小拳石（岩石系）
            enemy_level=12
        )

        prev_state = BattleState(
            phase=BattlePhase.BATTLE_START,
            player_hp=45,
            enemy_hp=40,
            player_level=13,
            enemy_level=12,
            player_pokemon_id=1,
            enemy_pokemon_id=74
        )

        # 可用招式
        available_moves = ["TACKLE", "GROWL", "VINE_WHIP"]

        turn = 0
        max_turns = 10

        while turn < max_turns:
            turn += 1

            # 获取AI推荐的最优动作
            optimal_move, optimal_score = action_evaluator.get_optimal_action(
                prev_state,
                available_moves
            )

            print(f"\n回合 {turn}:")
            print(f"  玩家HP: {prev_state.player_hp}/45 | 敌方HP: {prev_state.enemy_hp}/40")
            print(f"  AI选择: {optimal_move} (评分: {optimal_score:.3f})")

            # 模拟伤害（真实游戏中从内存读取）
            # ✅ 藤鞭对岩石系有克制效果（4倍）
            if optimal_move == "VINE_WHIP":
                damage = np.random.randint(15, 22)  # 高伤害
                action_desc = "🌿 藤鞭！（草系克制岩石）"
            elif optimal_move == "TACKLE":
                damage = np.random.randint(4, 8)    # 普通伤害
                action_desc = "💢 撞击"
            elif optimal_move == "GROWL":
                damage = 0  # 状态技能
                action_desc = "😾 叫声（降低攻击）"
            else:
                damage = 0
                action_desc = optimal_move

            # 更新状态
            curr_state = BattleState(
                phase=BattlePhase.PLAYER_TURN,
                player_hp=max(0, prev_state.player_hp - np.random.randint(3, 7)),
                enemy_hp=max(0, prev_state.enemy_hp - damage),
                player_level=13,
                enemy_level=12,
                player_pokemon_id=1,
                enemy_pokemon_id=74
            )

            # 计算奖励
            step_rewards = reward_calc.calculate_step_reward(
                curr_state,
                prev_state,
                optimal_move,
                optimal_move,
                optimal_score
            )

            total_reward = sum(step_rewards.values())

            print(f"  {action_desc}")
            print(f"  造成伤害: {damage}")
            print(f"  敌方HP: {prev_state.enemy_hp} → {curr_state.enemy_hp}")

            # 显示详细奖励
            if damage > 0:
                print(f"  💰 奖励明细:")
                for reward_name, reward_value in step_rewards.items():
                    if abs(reward_value) > 0.01:
                        print(f"     {reward_name}: {reward_value:+.3f}")
                print(f"  📊 总奖励: {total_reward:+.3f}")

            prev_state = curr_state

            # 检查战斗是否结束
            if curr_state.enemy_hp <= 0:
                curr_state.phase = BattlePhase.VICTORY
                print(f"\n🎉 胜利！")
                break

            if curr_state.player_hp <= 0:
                curr_state.phase = BattlePhase.DEFEAT
                print(f"\n💀 失败...")
                break

        # 终局奖励
        terminal_rewards = reward_calc.calculate_battle_terminal_reward(curr_state, turn)
        terminal_total = sum(terminal_rewards.values())

        print(f"\n🏁 战斗结束")
        print(f"  终局奖励: {terminal_total:+.3f}")
        for reward_name, reward_value in terminal_rewards.items():
            if abs(reward_value) > 0.01:
                print(f"    {reward_name}: {reward_value:+.3f}")

        battle_manager.end_battle(curr_state.phase == BattlePhase.VICTORY)

    # 输出最终学习结果
    print("\n" + "="*60)
    print("🎓 学习结果分析")
    print("="*60)

    reward_calc.print_statistics()

    print("\n" + "="*60)
    print("💡 AI学习到的知识")
    print("="*60)

    move_stats = action_evaluator.get_move_statistics()

    if "VINE_WHIP" in move_stats and "TACKLE" in move_stats:
        vine_whip_dmg = move_stats["VINE_WHIP"]["avg_damage"]
        tackle_dmg = move_stats["TACKLE"]["avg_damage"]
        ratio = vine_whip_dmg / max(tackle_dmg, 0.01)

        print(f"藤鞭平均伤害: {vine_whip_dmg:.1%} HP")
        print(f"撞击平均伤害: {tackle_dmg:.1%} HP")
        print(f"效率比: {ratio:.1f}x")
        print(f"\n✅ AI学会了：藤鞭比撞击强 {ratio:.1f} 倍！")
        print(f"✅ 这正是草系克制岩石系的体现（4倍克制）")

    print("\n" + "="*60)
    print("🚀 后续优化建议")
    print("="*60)
    print("1. 整合到主训练循环")
    print("2. 保存学习到的招式效果（持久化）")
    print("3. 针对不同敌人使用不同策略")
    print("4. 添加ε-greedy探索（偶尔尝试新招式）")