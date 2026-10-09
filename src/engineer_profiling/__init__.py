# -*- coding: utf-8 -*-
"""
伙伴工程师能力画像与培训推荐系统
================================

面向 to B 交付业务的能力管理工具：

- **画像构建**：基于历史工单 / 培训记录 / 认证考试，为每位工程师构建技能能力画像
- **聚类分析**：把工程师按能力向量聚成技能群组，从"全员一锅端"走向"分组精准推"
- **个性推荐**：根据能力短板，为每位工程师召回并排序个性化培训内容
- **报告输出**：生成每人雷达表 + 团队热力矩阵 + 群组培训建议

本项目全部用本地规则 / 统计 / 无监督机器学习实现，**不依赖任何真实 LLM API**，
离线即可跑通，测试秒级完成。

版本号：0.1.0
"""

__version__ = "0.1.0"

from .data import (
    Engineer,
    TicketRecord,
    TrainingRecord,
    CertificationRecord,
    PRODUCT_LINES,
    DIFFICULTY_LEVELS,
    classify_ticket,
    generate_synthetic_team,
)
from .profiling import AbilityBuilder
from .clustering import SkillClusterer
from .recommend import Course, Recommender
from .report import ReportWriter
from .pipeline import run_pipeline

__all__ = [
    "Engineer",
    "TicketRecord",
    "TrainingRecord",
    "CertificationRecord",
    "PRODUCT_LINES",
    "DIFFICULTY_LEVELS",
    "classify_ticket",
    "generate_synthetic_team",
    "AbilityBuilder",
    "SkillClusterer",
    "Course",
    "Recommender",
    "ReportWriter",
    "run_pipeline",
]