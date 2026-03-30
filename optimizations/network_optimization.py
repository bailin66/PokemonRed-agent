# -*- coding: utf-8 -*-
class ImprovedPolicyNetwork:
    @staticmethod
    def suggest_architecture():
        return {
            "input_layers": ["game_state", "battle_state"],
            "hidden_units": [256, 256],
            "output_dim": 12
        }
