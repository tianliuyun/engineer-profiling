# -*- coding: utf-8 -*-
"""
pipeline：一键编排 画像 -> 聚类 -> 推荐 -> 报告 的完整链路
=========================================================

供 ``examples/demo.py`` 与测试使用，把四个模块串成一条可复现的流水线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from . import __version__
from .profiling import AbilityBuilder
from .clustering import SkillClusterer
from .recommend import Recommender
from .report import ReportWriter
from .data import Engineer, generate_synthetic_team


@dataclass
class PipelineResult:
    """流水线的一次完整输出。"""

    engineers: List[Engineer]
    clusters: SkillClusterer
    builder: AbilityBuilder
    recommender: Recommender
    writer: ReportWriter
    recommends: Dict[str, List[Dict]]
    report: Dict


def run_pipeline(
    n_engineers: int = 24,
    seed: int = 42,
    n_clusters: int = 3,
    top_k: int = 3,
) -> PipelineResult:
    """运行整条能力画像 -> 分组 -> 推荐 -> 报告链路。"""
    team = generate_synthetic_team(n_engineers=n_engineers, seed=seed)
    builder = AbilityBuilder()
    clusters = SkillClusterer(n_clusters=n_clusters, seed=seed, builder=builder).fit(team)
    recommender = Recommender(builder=builder, top_k=top_k)
    recommends = recommender.recommend_many(team)
    writer = ReportWriter(builder=builder)
    report = writer.build_package(team, clusters, recommender)
    return PipelineResult(
        engineers=team,
        clusters=clusters,
        builder=builder,
        recommender=recommender,
        writer=writer,
        recommends=recommends,
        report=report,
    )