# -*- coding: utf-8 -*-
"""
聚类模块：把工程师按能力向量分组
================================

思想：画像矩阵摊成一维能力向量，用 **K-Means** 在能力空间里分组，
再结合每个群组的产品线均分，自动为群组取一个可读的名字（如"安全专精组"）。

价值点：培训从"全员一锅端"升级为"按技能群组分班授课"——
同一群组内能力相近，培训内容与难度才能统一、精准。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import Engineer, PRODUCT_LINES
from .profiling import AbilityBuilder

# 各产品线的中文"专精"命名
GROUP_NAMES = {
    "计算资源": "计算专精",
    "网络安全": "安全专精",
    "数据存储": "存储专精",
    "云原生": "云原生专精",
    "数据分析": "数据专精",
}


class SkillClusterer:
    """基于能力向量的工程师技能聚类。"""

    def __init__(
        self,
        n_clusters: int = 3,
        seed: int = 42,
        builder: AbilityBuilder | None = None,
    ):
        self.n_clusters = n_clusters
        self.seed = seed
        self.builder = builder or AbilityBuilder()
        self._model: KMeans | None = None
        self._scaler: StandardScaler | None = None
        self._labels: np.ndarray | None = None

    def feature_matrix(self, engineers: List[Engineer]) -> np.ndarray:
        """画像矩阵 -> 用于聚类的 (n_engineers, n_products*n_difficulty) 特征。"""
        rows = [self.builder.build_matrix(e).flatten() for e in engineers]
        return np.vstack(rows)

    def fit(self, engineers: List[Engineer]) -> "SkillClusterer":
        """训练聚类模型并保存分组结果。"""
        X = self.feature_matrix(engineers)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        self._labels = self._model.fit_predict(Xs)
        return self

    def labels(self) -> List[int]:
        """按工程师顺序返回所属分组 id。"""
        if self._labels is None:
            raise RuntimeError("请先调用 fit()")
        return [int(x) for x in self._labels]

    def group_summary(self, engineers: List[Engineer]) -> Tuple[List[Dict], np.ndarray]:
        """返回 (群组摘要列表, 分组标签数组)。

        每个群组摘要：{群组: id, 人数, 命名, 平均综合分, 强势产品线, 短板产品线}
        """
        if self._labels is None:
            raise RuntimeError("请先调用 fit()")
        mats = np.stack([self.builder.build_matrix(e) for e in engineers])
        overalls = mats.sum(axis=2)  # (n, n_products) 产品线综合分
        summary = []
        for g in range(self.n_clusters):
            members = [i for i, lab in enumerate(self._labels) if int(lab) == g]
            if not members:
                summary.append({
                    "群组": g, "人数": 0, "命名": f"空组{g}",
                    "平均综合分": 0.0, "强势产品线": "无", "短板产品线": "无",
                })
                continue
            gm = overalls[members, :].mean(axis=0)
            strong = PRODUCT_LINES[int(np.argmax(gm))]
            weak = PRODUCT_LINES[int(np.argmin(gm))]
            summary.append({
                "群组": g,
                "人数": len(members),
                "命名": GROUP_NAMES.get(strong, strong) + "组",
                "平均综合分": round(float(gm.mean()), 3),
                "强势产品线": strong,
                "短板产品线": weak,
            })
        return summary, self._labels