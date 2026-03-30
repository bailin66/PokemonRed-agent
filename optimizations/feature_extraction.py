"""
优化方案3：工程特征提取系统
原问题：状态空间表示不充分，仅使用RGB像素

改进方案：
1. 提取游戏引擎级别的关键特征
2. 构建结构化的状态表示
3. 结合像素和符号特征进行学习
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import struct

@dataclass
class EngineeringFeatures:
    """游戏引擎特征容器"""
    # 位置特征
    player_position: Tuple[int, int]
    current_map_id: int
    
    # 战斗特征
    in_battle: bool
    enemy_level: int
    enemy_hp: int
    player_hp: int
    
    # 库存特征
    inventory_items: Dict[str, int]
    pokemons: List[Dict]
    
    # 进度特征
    badges_earned: int
    pokemon_caught: int
    
    # UI特征
    is_in_menu: bool
    menu_state: str
    
    # 其他
    money: int
    poison_status: bool


class FeatureExtractor:
    """
    从游戏内存和像素中提取工程特征
    需要配合PyBoy的内存访问接口
    """
    
    # Pokemon Red的关键内存地址
    MEMORY_ADDRESSES = {
        "player_x": 0xD361,
        "player_y": 0xD362,
        "current_map": 0xD35E,
        "in_battle": 0xD057,
        "enemy_level": 0xCFF3,
        "enemy_hp": 0xCFF4,
        "player_hp": 0xD15E,
        "badges": 0xD356,
        "caught_count": 0xD2F7,
        "money_h": 0xD347,
        "money_l": 0xD348,
        "poison": 0xD2F8,
    }
    
    def __init__(self, pyboy_instance=None):
        """
        初始化特征提取器
        
        Args:
            pyboy_instance: PyBoy游戏实例
        """
        self.pyboy = pyboy_instance
        self.feature_history = []
        self.max_history = 50
    
    def read_memory(self, address: int) -> int:
        """读取内存"""
        if self.pyboy is not None:
            return self.pyboy.memory[address]
        return 0
    
    def extract_position_features(self) -> Dict:
        """提取位置特征"""
        x = self.read_memory(self.MEMORY_ADDRESSES["player_x"])
        y = self.read_memory(self.MEMORY_ADDRESSES["player_y"])
        map_id = self.read_memory(self.MEMORY_ADDRESSES["current_map"])
        
        return {
            "player_x": x,
            "player_y": y,
            "current_map": map_id,
            "position_hash": hash((map_id, x, y)) % 10000,  # 用于探索奖励
        }
    
    def extract_battle_features(self) -> Dict:
        """提取战斗特征"""
        in_battle = self.read_memory(self.MEMORY_ADDRESSES["in_battle"]) > 0
        enemy_level = self.read_memory(self.MEMORY_ADDRESSES["enemy_level"])
        enemy_hp = self.read_memory(self.MEMORY_ADDRESSES["enemy_hp"])
        player_hp = self.read_memory(self.MEMORY_ADDRESSES["player_hp"])
        
        return {
            "in_battle": in_battle,
            "enemy_level": enemy_level,
            "enemy_hp": enemy_hp,
            "player_hp": player_hp,
            "hp_ratio": player_hp / max(enemy_hp, 1),
        }
    
    def extract_progress_features(self) -> Dict:
        """提取进度特征"""
        badges = self.read_memory(self.MEMORY_ADDRESSES["badges"])
        caught = self.read_memory(self.MEMORY_ADDRESSES["caught_count"])
        money_h = self.read_memory(self.MEMORY_ADDRESSES["money_h"])        
        money_l = self.read_memory(self.MEMORY_ADDRESSES["money_l"])
        money = (money_h << 8) | money_l
        
        return {
            "badges_earned": bin(badges).count('1'),  # 统计比特位
            "badges_bitmask": badges,
            "pokemon_caught": caught,
            "money": money,
        }
    
    def extract_status_features(self) -> Dict:
        """提取状态特征"""
        poison = self.read_memory(self.MEMORY_ADDRESSES["poison"]) > 0
        
        return {
            "poison_status": poison,
        }
    
    def extract_all_features(self) -> Dict:
        """提取所有特征"""
        features = {
            **self.extract_position_features(),
            **self.extract_battle_features(),
            **self.extract_progress_features(),
            **self.extract_status_features(),
        }
        
        # 记录历史
        self.feature_history.append(features)
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)
        
        return features
    
    def get_feature_vector(self, features: Dict = None, normalize: bool = True) -> np.ndarray:
        """
        将特征转换为向量形式
        
        Args:
            features: 特征字典
            normalize: 是否归一化
            
        Returns:
            特征向量
        """
        if features is None:
            features = self.extract_all_features()
        
        # 定义特征顺序和范围
        feature_list = [
            ("player_x", 0, 20),
            ("player_y", 0, 18),
            ("current_map", 0, 250),
            ("in_battle", 0, 1),
            ("enemy_level", 0, 60),
            ("enemy_hp", 0, 255),
            ("player_hp", 0, 255),
            ("hp_ratio", 0, 1),
            ("badges_earned", 0, 8),
            ("pokemon_caught", 0, 151),
            ("poison_status", 0, 1),
            ("money", 0, 100000),
        ]
        
        vector = []
        for feature_name, min_val, max_val in feature_list:
            value = features.get(feature_name, 0)
            
            # 范围限制
            value = np.clip(value, min_val, max_val)
            
            if normalize:
                # 归一化到[0, 1]
                value = (value - min_val) / max(max_val - min_val, 1e-8)
            
            vector.append(value)
        
        return np.array(vector, dtype=np.float32)
    
    def extract_pixel_features(self, screen: np.ndarray) -> Dict:
        """
        从像素中提取额外特征
        
        Args:
            screen: RGB屏幕数据
            
        Returns:
            像素特征字典
        """
        # 计算屏幕的简单统计
        features = {
            "screen_brightness": np.mean(screen),
            "screen_contrast": np.std(screen),
            "screen_redness": np.mean(screen[:, :, 0]),
            "screen_greeness": np.mean(screen[:, :, 1]),
            "screen_blueness": np.mean(screen[:, :, 2]),
        }
        
        # 检测画面变化
        if len(self.feature_history) > 1:
            # 简单的变化检测
            prev_brightness = self.feature_history[-1].get("screen_brightness", 0)
            curr_brightness = features["screen_brightness"]
            features["brightness_change"] = abs(curr_brightness - prev_brightness)
        else:
            features["brightness_change"] = 0
        
        return features


class StateRepresentation:
    """
    结构化状态表示
    结合工程特征和原始像素
    """
    
    def __init__(self, feature_extractor: FeatureExtractor, pixel_size: Tuple[int, int] = (84, 84)):
        self.feature_extractor = feature_extractor
        self.pixel_size = pixel_size
        self.state_history = []
    
    def build_state(
        self,
        screen: np.ndarray,
        game_features: Dict = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建完整的状态表示
        
        Args:
            screen: 原始屏幕数据
            game_features: 游戏特征（可选）
            
        Returns:
            (processed_screen, feature_vector)
        """
        # 处理屏幕
        processed_screen = self._process_screen(screen)
        
        # 提取特征
        if game_features is None:
            game_features = self.feature_extractor.extract_all_features()
        
        # 获取特征向量
        feature_vector = self.feature_extractor.get_feature_vector(game_features, normalize=True)
        
        # 添加像素特征
        pixel_features = self.feature_extractor.extract_pixel_features(screen)
        pixel_features_vector = np.array([
            pixel_features["screen_brightness"],
            pixel_features["screen_contrast"],
            pixel_features["brightness_change"],
        ], dtype=np.float32)
        
        # 合并特征
        combined_features = np.concatenate([feature_vector, pixel_features_vector])
        
        return processed_screen, combined_features
    
    def _process_screen(self, screen: np.ndarray) -> np.ndarray:
        """处理屏幕数据"""
        # 调整大小
        screen_resized = np.array(
            Image.fromarray(screen).resize(self.pixel_size, Image.BILINEAR)
        )
        
        # 转换为灰度图（可选）
        # screen_gray = np.dot(screen_resized[...,:3], [0.2989, 0.5870, 0.1140])
        
        return screen_resized.astype(np.float32) / 255.0
    
    def get_state_encoding(self) -> np.ndarray:
        """获取状态编码用于网络输入"""
        if len(self.state_history) == 0:
            return np.zeros(512, dtype=np.float32)
        
        # 使用最近的特征
        recent_features = [s[1] for s in self.state_history[-4:]]  # 最近4个时间步
        
        # 拼接特征
        encoding = np.concatenate(recent_features) if recent_features else np.zeros(512, dtype=np.float32)
        
        return encoding


class AdaptiveFeatureSelector:
    """
    自适应特征选择器
    根据训练进度动态选择最重要的特征
    """
    
    def __init__(self, total_features: int, k: int = 10):
        """
        Args:
            total_features: 总特征数
            k: 选择的特征数
        """
        self.total_features = total_features
        self.k = k
        self.feature_importance = np.ones(total_features) / total_features
        self.update_count = 0
    
    def update_importance(self, feature_gradients: np.ndarray):
        """
        基于梯度更新特征重要性
        
        Args:
            feature_gradients: 特征的梯度
        """
        # 计算特征的绝对贡献度
        importance = np.abs(feature_gradients)
        
        # 指数移动平均
        alpha = 0.1
        self.feature_importance = (
            (1 - alpha) * self.feature_importance + 
            alpha * (importance / (np.sum(importance) + 1e-8))
        )
        
        self.update_count += 1
    
    def select_features(self, features: np.ndarray) -> np.ndarray:
        """
        选择最重要的特征
        
        Args:
            features: 原始特征向量
            
        Returns:
            选择的特征
        """
        top_k_indices = np.argsort(self.feature_importance)[-self.k:]
        return features[top_k_indices]
    
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        return self.feature_importance.copy()


# ============= 使用示例 =============

if __name__ == "__main__":
    print("="*60)
    print("工程特征提取系统演示")
    print("="*60)
    
    # 初始化特征提取器
    extractor = FeatureExtractor()
    
    # 模拟特征提取
    print("\n提取的特征:")
    features = extractor.extract_all_features()
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # 获取特征向量
    feature_vector = extractor.get_feature_vector(features, normalize=True)
    print(f"\n特征向量长度: {len(feature_vector)}")
    print(f"特征向量示例: {feature_vector[:5]}")
    
    # 测试自适应选择
    print("\n" + "="*60)
    print("自适应特征选择演示")
    print("="*60)
    
    selector = AdaptiveFeatureSelector(total_features=len(feature_vector), k=5)
    
    # 模拟多次更新
    for i in range(10):
        fake_gradients = np.random.randn(len(feature_vector))
        selector.update_importance(fake_gradients)
    
    importance = selector.get_feature_importance()
    print(f"特征重要性(前5): {importance[:5]}")
    
    selected = selector.select_features(feature_vector)
    print(f"选择的特征数: {len(selected)}")
