# -*- coding: utf-8 -*-
"""
数据层：数据模型 + 文本分类 + 合成数据生成
==========================================

本项目使用**合成数据**（Mock）来演示整条链路，保证：
1. 离线可用：不依赖任何外部 API
2. 可复现：固定随机种子，每次运行结果一致
3. 秒级跑通：测试与演示瞬间完成

文本分类是本项目的核心入口之一：工单标题是自由文本，我们需要把它自动打上
「产品线」与「难度」标签，才能进入后续的画像统计。这里实现了一个**基于词典的
多分类器**（关键词计数 + 打分取 argmax），并兼容规则兜底。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# 领域常量（通用 IT 能力分类，非任何厂商专有产品）
# ---------------------------------------------------------------------------
# fmt: off
PRODUCT_LINES: List[str] = [
    "计算资源",   # 云主机 / 弹性计算相关
    "网络安全",   # 安全加固 / 边界防护
    "数据存储",   # 对象存储 / 文件存储
    "云原生",     # 容器 / K8s / 微服务
    "数据分析",   # 大数据处理 / 报表
]
DIFFICULTY_LEVELS: List[str] = ["初级", "中级", "高级"]
# fmt: on

# 文本分类用的关键词词典：每个产品线一组可命中关键词
KEYWORDS: Dict[str, List[str]] = {
    "计算资源": ["实例", "弹性", "cpu", "内存", "扩容", "重启", "虚拟机", "镜像"],
    "网络安全": ["防火墙", "漏洞", "攻击", "安全", "加固", "入侵", "策略", "防篡改"],
    "数据存储": ["存储桶", "对象存储", "备份", "归档", "快照", "云盘", "共享目录"],
    "云原生": ["容器", "k8s", "kubernetes", "pod", "镜像仓库", "微服务", "helm", "部署"],
    "数据分析": ["报表", "sql", "数据清洗", "离线任务", "flink", "数仓", "etl", "出账"],
}

# 难度判定的"高级信号"关键词：命中越多，说明越复杂
HARD_KEYWORDS: List[str] = ["集群", "故障转移", "性能调优", "分布式", "迁移", "多可用区", "高可用", "内核"]

# 冷启动基准分：没有任何历史数据时的初始画像（来自入职测评/考试成绩兜底）
COLD_START_BASE = 0.35


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class TicketRecord:
    """一份历史工单处理记录。``product``/``difficulty`` 通常由文本分类自动打标，
    也可在数据导入阶段人工预打标。"""

    text: str
    product: str = ""
    difficulty: str = ""
    resolved: bool = True          # 是否一次解决
    transferred: bool = False      # 是否转单（越少越好）
    duration_hours: float = 1.0    # 解决时长
    satisfaction: float = 0.7      # 客户满意度 0~1


@dataclass
class TrainingRecord:
    """一条培训/学习记录。"""

    course_title: str
    product: str
    passed: bool = True
    score: float = 0.7            # 结课成绩 0~1


@dataclass
class CertificationRecord:
    """一条认证考试记录。"""

    name: str
    product: str
    passed: bool
    score: float = 0.0            # 未过时保留 0


@dataclass
class Engineer:
    """一位伙伴工程师的基础信息 + 历史行为数据。"""

    name: str
    years: float = 1.0
    tickets: List[TicketRecord] = field(default_factory=list)
    trainings: List[TrainingRecord] = field(default_factory=list)
    certifications: List[CertificationRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 文本分类：工单标题 -> (产品线, 难度)
# ---------------------------------------------------------------------------
def _hit_count(text: str, words: List[str]) -> int:
    low = text.lower()
    return sum(1 for w in words if w in low)


def classify_ticket(text: str) -> Dict[str, str]:
    """把一段工单文本分类为 (产品线, 难度)。

    - 产品线：统计每个产品线关键词的命中数，取命中最多者为产品；平票时按产品线
      字典序兜底，保证结果确定可复现。
    - 难度：统计"高级信号"关键词数量；>=2 记高级，==1 记中级，否则初级。
    - 兜底：文本过短或毫无命中时，落到 "计算资源 / 初级"，保证不抛异常。
    """
    text = (text or "").strip()
    product = "计算资源"
    best = -1
    for line, words in KEYWORDS.items():
        hit = _hit_count(text, words)
        if hit > best:
            best = hit
            product = line
    # 完全无命中兜底
    if best <= 0:
        product = "计算资源"

    hard = _hit_count(text, HARD_KEYWORDS)
    if hard >= 2:
        difficulty = "高级"
    elif hard == 1:
        difficulty = "中级"
    else:
        difficulty = "初级"

    return {"product": product, "difficulty": difficulty}


# ---------------------------------------------------------------------------
# 合成数据生成（Mock）
# ---------------------------------------------------------------------------
def _product_weights(seed: int) -> List[float]:
    rng = np.random.RandomState(seed)
    w = rng.dirichlet(np.ones(len(PRODUCT_LINES)) * 1.2)
    return list(w)


def generate_synthetic_team(
    n_engineers: int = 24,
    seed: int = 42,
    tickets_per_engineer: int = 40,
) -> List[Engineer]:
    """生成一支规模可调的合成伙伴工程师团队，用于演示完整链路。

    用不同 ``skill`` 参数构造三档水平人群，方便后续画像 / 聚类验证"能区分度"：
    - 前列 1/3：新手（整体能力低，短板分散）
    - 中列 1/3：熟手（各有 1~2 个突出短板）
    - 后列 1/3：专家（整体能力强）
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    team: List[Engineer] = []

    n_products = len(PRODUCT_LINES)
    for idx in range(n_engineers):
        # 三档基准水平
        band = idx // max(1, (n_engineers // 3))
        if band == 0:
            base = 0.30   # 新手
        elif band == 1:
            base = 0.55   # 熟手
        else:
            base = 0.78   # 专家

        # 每个产品线的真实水平：基准 + 噪声，含 1 个明显短板
        skill = np.clip(base + np_rng.normal(0, 0.12, n_products), 0.15, 0.95)
        weak = idx % n_products
        skill[weak] = max(0.15, base - 0.25)   # 人为制造短板

        tickets: List[TicketRecord] = []
        for _ in range(tickets_per_engineer):
            p = int(np_rng.randint(0, n_products))
            s = float(skill[p])
            # 水平越高：一次解决率越高、转单越少、时长越短、满意度越高
            resolved = np_rng.rand() < (0.55 + 0.4 * s)
            transferred = (not resolved) and np_rng.rand() < (0.8 - 0.6 * s)
            duration = float(np.clip(4.0 - 3.0 * s + np_rng.normal(0, 0.8), 0.3, 8.0))
            satisfaction = float(np.clip(0.45 + 0.4 * s + np_rng.normal(0, 0.08), 0.2, 1.0))
            resolved = bool(resolved)
            # 部分工单标题可命中产品关键词，用于文本分类演示
            words = KEYWORDS[PRODUCT_LINES[p]]
            joined = f"{PRODUCT_LINES[p]}相关工单 {rng.choice(words)} 出现异常"
            tickets.append(
                TicketRecord(
                    text=joined,
                    product=PRODUCT_LINES[p],
                    difficulty=(
                        "高级" if resolved and s > 0.7 else
                        "初级" if s < 0.4 else "中级"
                    ),
                    resolved=resolved,
                    transferred=transferred,
                    duration_hours=duration,
                    satisfaction=satisfaction,
                )
            )

        # 培训 + 认证（带一部分未通过，用于短板回填）
        trainings: List[TrainingRecord] = []
        certifications: List[CertificationRecord] = []
        for p_idx in range(n_products):
            s = float(skill[p_idx])
            prod = PRODUCT_LINES[p_idx]
            passed = np_rng.rand() < (0.4 + 0.5 * s)
            trainings.append(
                TrainingRecord(
                    course_title=f"{prod}基础课程",
                    product=prod,
                    passed=passed,
                    score=float(np.clip(s + np_rng.normal(0, 0.1), 0, 1.0)),
                )
            )
            cert_passed = np_rng.rand() < (0.2 + 0.5 * s)
            certifications.append(
                CertificationRecord(
                    name=f"{prod}认证考试",
                    product=prod,
                    passed=cert_passed,
                    score=float(s) if cert_passed else 0.0,
                )
            )

        team.append(
            Engineer(
                name=f"工程师{idx + 1:02d}",
                years=round(1.0 + base * 5 + np_rng.uniform(0, 1), 1),
                tickets=tickets,
                trainings=trainings,
                certifications=certifications,
            )
        )

    return team