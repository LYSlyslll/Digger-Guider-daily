"""Stub script kept for backward compatibility.

This project now trains exclusively on 日频(日线)特征，不再生成或使用
15 分钟聚合的高频数据。为了避免误调用旧的重采样逻辑，本脚本
在运行时直接抛出异常并给出指引。
"""

raise RuntimeError(
    "High-frequency resampling is disabled. The pipeline now consumes only "
    "daily features; please use day_csi300.pkl directly."
)
