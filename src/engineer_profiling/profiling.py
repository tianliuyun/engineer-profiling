# -*- coding: utf-8 -*-
"""
画像构建模块：把历史行为数据压成每个人的能力向量
================================================

能力画像由三路信号融合而成（权重可配），每路都归一化到 0~1：

1. **工单能力**（权重 0.6，最重要）：以产品线为单位，把一次解决率、转单率(反向)、
   解决时长(反向)、客户满意度聚合成一个 0~1 分数。量越大越有统计意义，票数少的
   特征自动向先验回调，避免"只处理过 2 单就判满分"。
2. **培训掌握**（权重 0.2）：该产品线课程的通过率与结课成绩。
3. **认证水平**（权重 0.2）：该产品线认证考试是否通过及分数。

冷启动：工程师在某个产品线没有任何记录时，用 ``COLD_START_BASE`` 兜底，
后续随数据积累逐步被真实能力替代。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from .data import COLD_START_BASE, DIFFICULTY_LEVELS, Engineer, PRODUCT_LINES

# 默认三路信号权重（工单 > 培训 = 认证）
DEFAULT_WEIGHTS = {"ticket": 0.6, "training": 0.2, "certification": 0.2}


class AbilityBuilder:
    """从工程师的历史数据构建 (产品线 x 难度) 二维能力矩阵，并输出可读画像。"""

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        prior: float = COLD_START_BASE,
        min_samples: int = 5,
    ):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.prior = prior
        self.min_samples = min_samples

    # ------------------------------------------------------------------
    # 工单能力
    # ------------------------------------------------------------------
    def _ticket_score(self, engineer: Engineer, product: str) -> float:
        """聚合某产品线下该工程师所有工单的 4 个子指标 -> 0~1。"""
        tickets = [t for t in engineer.tickets if t.product == product]
        n = len(tickets)
        if n == 0:
            return self.prior

        resolve = np.mean([1.0 if t.resolved else 0.0 for t in tickets])
        # 转单越多越差 -> 用 (1 - 转单率)
        transfer = np.mean([1.0 if t.transferred else 0.0 for t in tickets])
        # 解决时长：经验上 1h 内满分，6h 以上归零，线性映射
        duration = np.mean([t.duration_hours for t in tickets])
        speed = float(np.clip(1.0 - (duration - 1.0) / 5.0, 0.0, 1.0))
        satis = np.mean([t.satisfaction for t in tickets])

        raw = float(np.mean([resolve, 1.0 - transfer, speed, satis]))
        # 样本量太少时向先验回调，防止幸存者偏差
        alpha = n / (n + self.min_samples)
        return alpha * raw + (1 - alpha) * self.prior

    # ------------------------------------------------------------------
    # 培训 / 认证
    # ------------------------------------------------------------------
    def _training_score(self, engineer: Engineer, product: str) -> float:
        recs = [r for r in engineer.trainings if r.product == product]
        if not recs:
            return self.prior
        mean_score = float(np.mean([r.score for r in recs])) if any(r.score > 0 for r in recs) else 0.0
        pass_rate = float(np.mean([1.0 if r.passed else 0.0 for r in recs]))
        return float(np.mean([pass_rate, mean_score]))

    def _certification_score(self, engineer: Engineer, product: str) -> float:
        recs = [r for r in engineer.certifications if r.product == product]
        if not recs:
            return self.prior
        has_pass = float(np.mean([1.0 if r.passed else 0.0 for r in recs]))
        best = max((r.score for r in recs if r.passed), default=0.0)
        return float(np.mean([has_pass, best]))

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def build_matrix(self, engineer: Engineer) -> np.ndarray:
        """返回 shape = (n_products, n_difficulty) 的能力矩阵，元素 0~1。

        difficulty 维度体现"同一产品线下工程师能扛住的难度上限"：
        若某产品线高分工单占比较高，则该难度档的"能力密度"更高，用于后续派生
        单产品线综合分时做难度加成。
        """
        matrix = np.zeros((len(PRODUCT_LINES), len(DIFFICULTY_LEVELS)))
        for p_idx, product in enumerate(PRODUCT_LINES):
            ticket = self._ticket_score(engineer, product)
            training = self._training_score(engineer, product)
            cert = self._certification_score(engineer, product)
            overall = (
                self.weights["ticket"] * ticket
                + self.weights["training"] * training
                + self.weights["certification"] * cert
            )
            # 按各难度档工单占比把 overall 摊到难度维度，突出"能扛多难"
            tickets = [t for t in engineer.tickets if t.product == product]
            if tickets:
                counts = np.zeros(len(DIFFICULTY_LEVELS))
                for t in tickets:
                    counts[DIFFICULTY_LEVELS.index(t.difficulty) if t.difficulty in DIFFICULTY_LEVELS else 1] += 1
                dist = counts / counts.sum()
                matrix[p_idx, :] = overall * dist
            else:
                matrix[p_idx, :] = overall / len(DIFFICULTY_LEVELS)
        return matrix

    def profile(self, engineer: Engineer) -> Dict[str, Dict[str, float]]:
        """生成人读的画像：{产品线: {难度: 分数, '综合': 分数}}，并附总均分。"""
        matrix = self.build_matrix(engineer)
        out: Dict[str, Dict[str, float]] = {}
        overall = matrix.sum(axis=1)   # 每产品线综合分（难度维求和）
        total = float(matrix.sum())
        for i, product in enumerate(PRODUCT_LINES):
            row = {
                DIFFICULTY_LEVELS[j]: round(float(matrix[i, j]), 3)
                for j in range(len(DIFFICULTY_LEVELS))
            }
            row["综合"] = round(overall[i], 3)
            out[product] = row
        return {
            "工程师": engineer.name,
            "工龄": engineer.years,
            "平均综合分": round(total / (len(PRODUCT_LINES) * len(DIFFICULTY_LEVELS)), 3),
            "画像": out,
        }