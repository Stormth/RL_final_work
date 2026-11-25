"""rl_algo：执行反馈驱动的代码修复强化学习。

分层：

- ``rl_algo.env``   受限执行、逐 assert 判定、提示词构造（零依赖，纯标准库）
- ``rl_algo.data``  题库拉取、清洗、时间切分、难度校准
- ``rl_algo.train`` GRPO/GDPO 训练与奖励函数（需 GPU 与 trl）
"""

__version__ = "0.1.0"
