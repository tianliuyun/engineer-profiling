# -*- coding: utf-8 -*-
"""engineer-profiling 项目测试集。

覆盖四类能力点，共 14 个用例：
1. 文本分类（词典打标）
2. 画像构建（能力矩阵）
3. 聚类分析（分组可区分）
4. 个性化推荐 + 报告输出 + 流水线端到端
"""
import numpy as np
import pytest

from engineer_profiling import (
    AbilityBuilder,
    CertificationRecord,
    Course,
    DIFFICULTY_LEVELS,
    Engineer,
    PRODUCT_LINES,
    Recommender,
    ReportWriter,
    SkillClusterer,
    TicketRecord,
    TrainingRecord,
    classify_ticket,
    generate_synthetic_team,
    run_pipeline,
)


@pytest.fixture(scope="module")
def team():
    return generate_synthetic_team(n_engineers=24, seed=42)


# ===========================================================================
# 1. 文本分类
# ===========================================================================
class TestTextClassification:
    def test_classify_product_line(self):
        """工单标题含产品关键词时应命中对应产品线。"""
        r = classify_ticket("防火墙出现漏洞，需要安全加固处理")
        assert r["product"] == "网络安全"

        r2 = classify_ticket("对象存储桶备份失败，快照无法归档")
        assert r2["product"] == "数据存储"

    def test_classify_difficulty_by_hard_signals(self):
        """命中多个"高级信号"关键词时判为高级难度。"""
        r = classify_ticket("集群故障转移失败，需做分布式迁移与性能调优、多可用区高可用")
        assert r["difficulty"] == "高级"

        r2 = classify_ticket("实例无法重启")
        assert r2["difficulty"] == "初级"

    def test_classify_empty_fallback(self):
        """空文本/无命中必须有确定性兜底，不抛异常。"""
        assert classify_ticket("") == {"product": "计算资源", "difficulty": "初级"}
        assert classify_ticket("!!!")["product"] == "计算资源"


# ===========================================================================
# 2. 画像构建
# ===========================================================================
class TestAbilityBuilder:
    def test_engineer_shapes(self, team):
        """每个工程师都要产出 (产品线数 x 难度数) 的能力矩阵，且全部归一化到 0~1。"""
        builder = AbilityBuilder()
        for e in team[:5]:
            mat = builder.build_matrix(e)
            assert mat.shape == (len(PRODUCT_LINES), len(DIFFICULTY_LEVELS))
            assert mat.min() >= 0.0 and mat.max() <= 1.0

    def test_cold_start_prior(self):
        """没有任何历史数据的工程师，画像应回落并保持稳定的冷启动基准分。"""
        blank = Engineer(name="新人", years=0.0)
        builder = AbilityBuilder(prior=0.35)
        mat = builder.build_matrix(blank)
        # 全矩阵都是基准分 / 难度档数，应等于 0.35/3
        expected = 0.35 / 3
        assert np.allclose(mat, expected, atol=1e-6)

    def test_more_tickets_boosts_score(self):
        """只处理过高难度/高质量工单的工程师，在该产品线得分应显著更高。"""
        builder = AbilityBuilder()
        weak = Engineer(
            name="弱",
            tickets=[TicketRecord(text="x", product="网络安全", resolved=False,
                                  transferred=True, duration_hours=7.0, satisfaction=0.3)] * 20,
        )
        strong = Engineer(
            name="强",
            tickets=[TicketRecord(text="x", product="网络安全", resolved=True,
                                  transferred=False, duration_hours=1.0, satisfaction=1.0)] * 20,
        )
        wm = builder.build_matrix(weak).sum()
        sm = builder.build_matrix(strong).sum()
        assert sm > wm

    def test_build_profile_readable(self, team):
        """profile() 应产出人读的中文画像字典。"""
        prof = AbilityBuilder().profile(team[0])
        assert prof["工程师"] == team[0].name
        assert "平均综合分" in prof
        assert set(prof["画像"].keys()) == set(PRODUCT_LINES)
        # 每个产品的难度维 + 综合 都在
        row0 = prof["画像"][PRODUCT_LINES[0]]
        assert set(row0.keys()) == {"初级", "中级", "高级", "综合"}


# ===========================================================================
# 3. 聚类分析
# ===========================================================================
class TestSkillClusterer:
    def test_cluster_assignment(self, team):
        """所有工程师都必须被分到 [0, n_clusters) 中的某个组。"""
        clusterer = SkillClusterer(n_clusters=3, seed=42).fit(team)
        labels = clusterer.labels()
        assert len(labels) == len(team)
        assert set(labels) <= {0, 1, 2}

    def test_group_summary_structure(self, team):
        """每个群组摘要都应包含人数 / 命名 / 强势与短板产品线。"""
        clusterer = SkillClusterer(n_clusters=3, seed=42).fit(team)
        summary, _ = clusterer.group_summary(team)
        assert len(summary) == 3
        for g in summary:
            assert g["人数"] > 0
            assert g["命名"].endswith("组")
            assert g["强势产品线"] in PRODUCT_LINES
            assert g["短板产品线"] in PRODUCT_LINES
        # 人数之和应等于工程师总数
        assert sum(g["人数"] for g in summary) == len(team)


# ===========================================================================
# 4. 个性化推荐
# ===========================================================================
class TestRecommender:
    def test_recommend_returns_ranked_courses(self):
        """推荐应返回按短板优先排序的课程，且每门课程属真实课程库。"""
        builder = AbilityBuilder()
        team = generate_synthetic_team(n_engineers=4, seed=7)
        rec = Recommender(builder=builder, top_k=3)
        for e in team:
            items = rec.recommend(e)
            assert 1 <= len(items) <= 3
            titles = [i["课程"] for i in items]
            from engineer_profiling.recommend import COURSE_LIBRARY
            lib_titles = {c.title for c in COURSE_LIBRARY}
            assert all(t in lib_titles for t in titles)
            assert "推荐理由" in items[0]

    def test_recommend_order_prefers_weak(self):
        """短板越大的产品线应越靠前推荐。"""
        from engineer_profiling import PRODUCT_LINES
        # 构造一个网络安全能力明显最弱、其余很强的工程师
        builder = AbilityBuilder()
        e = Engineer(
            name="测试员",
            tickets=[
                TicketRecord(text="x", product="网络安全", resolved=False,
                             transferred=True, duration_hours=8.0, satisfaction=0.2),
                TicketRecord(text="x", product="计算资源", resolved=True,
                             transferred=False, duration_hours=1.0, satisfaction=0.95),
                TicketRecord(text="x", product="数据存储", resolved=True,
                             transferred=False, duration_hours=1.1, satisfaction=0.9),
                TicketRecord(text="x", product="云原生", resolved=True,
                             transferred=False, duration_hours=1.2, satisfaction=0.9),
                TicketRecord(text="x", product="数据分析", resolved=True,
                             transferred=False, duration_hours=1.3, satisfaction=0.88),
            ] * 30,
        )
        items = Recommender(builder=builder, top_k=3).recommend(e)
        assert items[0]["产品线"] == "网络安全"


# ===========================================================================
# 5. 报告输出 + 端到端流水线
# ===========================================================================
class TestReportAndPipeline:
    def test_report_build_package(self, team):
        """完整打包报告应含工程师表、群组表、冷启动统计。"""
        builder = AbilityBuilder()
        clusterer = SkillClusterer(n_clusters=3, seed=42, builder=builder).fit(team)
        recommender = Recommender(builder=builder)
        writer = ReportWriter(builder=builder)
        pkg = writer.build_package(team, clusterer, recommender)
        assert len(pkg["engineers"]) == len(team)
        assert len(pkg["groups"]) == 3
        assert pkg["cold_start"]["工程师总数"] == len(team)

    def test_csv_roundtrip(self, team):
        """工程师表应能导出为可解析的 CSV（含表头）。"""
        builder = AbilityBuilder()
        clusterer = SkillClusterer(n_clusters=2, seed=3, builder=builder).fit(team)
        recommender = Recommender(builder=builder)
        writer = ReportWriter(builder=builder)
        table = writer.engineer_table(team, recommender.recommend_many(team))
        csv_str = writer.to_csv(table)
        first_line = csv_str.splitlines()[0]
        assert "工程师" in first_line and "平均分" in first_line
        assert len(csv_str.splitlines()) == len(team) + 1

    def test_end_to_end_pipeline(self):
        """run_pipeline 应串起整条链路并返回结构化结果。"""
        result = run_pipeline(n_engineers=12, seed=9, n_clusters=3, top_k=3)
        assert len(result.report["engineers"]) == 12
        assert result.report["groups"]
        assert len(result.recommends) == 12
        # 断言至少存在一条真正的推荐
        assert any(recs for recs in result.recommends.values())

    def test_deterministic_seed(self):
        """相同种子应产出相同的画像与分组（可复现性）。"""
        r1 = run_pipeline(n_engineers=10, seed=100, n_clusters=2)
        r2 = run_pipeline(n_engineers=10, seed=100, n_clusters=2)
        assert [g["人数"] for g in r1.report["groups"]] == [g["人数"] for g in r2.report["groups"]]
        assert r1.report["engineers"] == r2.report["engineers"]