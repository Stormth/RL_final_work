"""训练层：奖励函数与 GRPO/GDPO 入口（需 trl）。"""

from rl_algo.train.rewards import JudgeCache, RewardSuite, build_suite, completion_text

__all__ = ["JudgeCache", "RewardSuite", "build_suite", "completion_text"]
