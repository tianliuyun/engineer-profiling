# -*- coding: utf-8 -*-
"""
报告输出模块：把画像 / 聚类 / 推荐整理成人类可读的结构化报告
============================================================

输出三部分，都可方便地再落成 CSV / Markdown / 表格：
- 工程师总览表（含冷启动人数统计）
- 团队技能分组表（聚类结果）
- 每人个性化培训推荐表
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from typing import Dict, List

from . import __version__
from .profiling import AbilityBuilder
from .clustering import SkillClusterer
from .recommend import Recommender
from .data import Engineer, PRODUCT_LINES


class ReportWriter:
    """把整条链路的产出整理成报告结构，并提供导出函数。"""

    def __init__(self, builder: AbilityBuilder | None = None):
        self.builder = builder or AbilityBuilder()

    def engineer_table(
        self,
        engineers: List[Engineer],
        recommends: Dict[str, List[Dict]],
    ) -> List[Dict]:
        """每位工程师一行：产品线平均分、强势/短板产品线、Top1 推荐。"""
        rows = []
        for e in engineers:
            prof = self.builder.profile(e)
            overall = prof["画像"]
            avg = prof["平均综合分"]
            line_scores = [overall[p]["综合"] for p in PRODUCT_LINES]
            strong = PRODUCT_LINES[int(line_scores.index(max(line_scores)))]
            weak = PRODUCT_LINES[int(line_scores.index(min(line_scores)))]
            recs = recommends.get(e.name, [])
            top1 = recs[0]["课程"] if recs else "无需补课"
            rows.append({
                "工程师": e.name,
                "工龄": e.years,
                "平均分": avg,
                "强势产品线": strong,
                "短板产品线": weak,
                "Top1推荐": top1,
                "推荐数": len(recs),
            })
        return rows

    def cold_start_stats(self, engineers: List[Engineer]) -> Dict[str, int]:
        """统计冷启动覆盖情况：任一产品线无任何工单记录，则该产品线用基准分兜底，
        画像随时间与数据积累逐步被真实能力替代。"""
        no_data_count = 0
        flagged_engineers = []
        for e in engineers:
            untouched = [
                p for p in PRODUCT_LINES
                if not any(t.product == p for t in e.tickets)
            ]
            if untouched:
                no_data_count += 1
                flagged_engineers.append(e.name)
        return {
            "工程师总数": len(engineers),
            "存在冷启动空洞的工程师数": no_data_count,
            "冷启动工程师示例": flagged_engineers[:3],
        }

    def group_table(self, clusters: SkillClusterer, engineers: List[Engineer]) -> List[Dict]:
        summary, _ = clusters.group_summary(engineers)
        return summary

    def to_csv(self, table: List[Dict]) -> str:
        """把一张表格转成 UTF-8 CSV 字符串（方便直接落盘 / Excel 打开）。"""
        if not table:
            return ""
        fieldnames = list(table[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
        return buf.getvalue()

    def build_package(
        self,
        engineers: List[Engineer],
        clusters: SkillClusterer,
        recommender: Recommender,
    ) -> Dict:
        """生成完整打包报告（JSON 结构，便于二次加工）。"""
        recommends = recommender.recommend_many(engineers)
        return {
            "shishen_version": __version__,
            "engineers": self.engineer_table(engineers, recommends),
            "groups": self.group_table(clusters, engineers),
            "cold_start": self.cold_start_stats(engineers),
        }