# -*- coding: utf-8 -*-
"""
推荐模块：基于短板召回 + 排序的个性化培训推荐
=============================================

经典推荐范式落地到培训场景，分两步：

1. **召回（Recall）**：能力越弱的产品线越需要补课，取短板 / 认证未过 / 未结课课程
   作为候选，控制候选集规模。
2. **排序（Rank）**：一个综合打分 ``score`` 决定推荐顺序——
   ``score = 短板程度(权重大) + 课程适配度(难度匹配) - 已掌握惩罚``。

输出附带一句中文结论，可直接写入报告：课程难度与"该产品线能力 < 0.4"对齐时，
推荐"基础班"；能力在 0.4~0.7 推荐"进阶班"；否则"高阶班/认证冲刺"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .data import Engineer, PRODUCT_LINES
from .profiling import AbilityBuilder


@dataclass
class Course:
    """一门可推荐的培训课程。"""

    title: str
    product: str
    level: str          # 基础班 / 进阶班 / 高阶班
    difficulty: int = 1  # 1=基础 2=进阶 3=高阶


# 一个覆盖全产品线 x 三档难度的课程库（演示用）
COURSE_LIBRARY: List[Course] = [
    Course(f"{p}基础班", p, "基础班", 1)
    for p in PRODUCT_LINES
] + [
    Course(f"{p}进阶班", p, "进阶班", 2)
    for p in PRODUCT_LINES
] + [
    Course(f"{p}高阶认证冲刺", p, "高阶班", 3)
    for p in PRODUCT_LINES
]


def _level_for(score: float) -> "tuple[str, int]":
    """按当前能力水平返回应推荐的课程难度。能力越低越要打基础。"""
    if score < 0.4:
        return "基础班", 1
    if score < 0.7:
        return "进阶班", 2
    return "高阶班", 3


class Recommender:
    """基于能力短板的培训内容推荐器。"""

    def __init__(self, builder: AbilityBuilder | None = None, top_k: int = 3):
        self.builder = builder or AbilityBuilder()
        self.top_k = top_k

    def _matrix(self, engineer: Engineer) -> np.ndarray:
        return self.builder.build_matrix(engineer)  # (n_products, n_difficulty)

    def recommend(self, engineer: Engineer) -> List[Dict]:
        """为单名工程师生成按优先级排序的培训推荐。"""
        mat = self._matrix(engineer)
        # 每产品线综合分（难度维求和）
        overall = mat.sum(axis=1)  # (n_products,)
        # 短板程度：越接近 0 越需要补课
        gap = 1.0 - np.clip(overall, 0.0, 1.0)

        products = np.argsort(-gap)  # 短板大的产品线排前面

        ranked: List[Dict] = []
        for p_idx in products:
            product = PRODUCT_LINES[int(p_idx)]
            score_here = float(overall[int(p_idx)])
            level, diff = _level_for(score_here)

            # 候选：该产品线下难度匹配的课程，最多 2 门
            candidates = [
                c for c in COURSE_LIBRARY
                if c.product == product and c.level == level
            ]
            for c in candidates[:1]:
                # 排序分：短板权重 0.7 + 难度适配 0.3，减去已掌握信号
                gap_score = gap[int(p_idx)]
                match = 1.0 if c.difficulty == diff else 0.5
                rank = 0.7 * gap_score + 0.3 * match
                ranked.append({
                    "课程": c.title,
                    "产品线": product,
                    "课程难度": c.level,
                    "短板程度": round(gap_score, 3),
                    "能力分": round(score_here, 3),
                    "推荐理由": (
                        f"{product}能力较弱（{score_here:.2f}），"
                        f"建议从{c.level}打基础到实战"
                    ),
                    "_sort": round(rank, 4),
                })

        ranked.sort(key=lambda r: r["_sort"], reverse=True)
        # 去重：同一产品线只保留一条，避免推荐卡被同一产品占满
        seen, top = set(), []
        for r in ranked:
            if r["产品线"] not in seen:
                seen.add(r["产品线"])
                top.append(r)
        return top[: self.top_k]

    def recommend_many(self, engineers: Sequence[Engineer]) -> Dict[str, List[Dict]]:
        """为多人为推荐，返回 {工程师名: [推荐...]}。"""
        return {e.name: self.recommend(e) for e in engineers}