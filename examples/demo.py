# -*- coding: utf-8 -*-
"""
演示脚本：一键跑通「伙伴工程师能力画像与培训推荐」完整链路
==========================================================

用法（在仓库根目录，已激活 venv）：
    python examples/demo.py

输出：
    - 画像构建（每人能力矩阵 / 平均分）
    - 聚类结果（技能群组 + 命名 + 强势/短板产品线）
    - 个性化培训推荐（每人 Top3）
    - 结构化打包报告（JSON，落盘演示）
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engineer_profiling import run_pipeline  # noqa: E402


def main() -> None:
    result = run_pipeline(n_engineers=24, seed=42, n_clusters=3, top_k=3)

    print("=" * 64)
    print("伙伴工程师能力画像与培训推荐系统 - 演示")
    print(f"工程师数量：{len(result.engineers)}  技能群组：{len(result.report['groups'])} 个")
    print("=" * 64)

    print("\n【一】画像构建（前 5 名工程师平均分 & 短板）")
    for row in result.report["engineers"][:5]:
        print(
            f"  {row['工程师']:<10} 平均 {row['平均分']:.2f}  "
            f"强项 {row['强势产品线']}  短板 {row['短板产品线']}  "
            f"推荐 {row['Top1推荐']}"
        )

    print("\n【二】团队技能聚类分组")
    for g in result.report["groups"]:
        print(
            f"  组{g['群组']} {g['命名']:<12} 人数 {g['人数']:>2}  "
            f"均分 {g['平均综合分']:.3f}  强在 {g['强势产品线']}  弱在 {g['短板产品线']}"
        )

    print("\n【三】个性化培训推荐（每人 Top3）")
    for name, recs in list(result.recommends.items())[:6]:
        print(f"  {name}:")
        for r in recs[:2]:
            print(f"     - {r['课程']}  |  短板 {r['短板程度']:.2f}  |  {r['推荐理由']}")

    print("\n【四】冷启动统计")
    cs = result.report["cold_start"]
    print(f"  工程师总数 {cs['工程师总数']} 人，存在冷启动空洞 {cs['存在冷启动空洞的工程师数']} 人")
    print(f"  冷启动示例：{cs['冷启动工程师示例']}")

    # 落盘报告
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已导出：{os.path.abspath(out_path)}")
    print("\n完成")


if __name__ == "__main__":
    main()