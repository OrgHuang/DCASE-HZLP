#!/usr/bin/env python3
"""
Logits-Confusion 交叉分析：
用 Yes-logit 分数来验证混淆假设 —— 即对于混淆型错误，
正确答案和模型选中的错误答案的 Yes-logit 分数应该非常接近。
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def char_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def token_overlap(a, b):
    toks_a = set(re.findall(r'\w+', a.lower()))
    toks_b = set(re.findall(r'\w+', b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def combined_similarity(a, b):
    return (token_overlap(a, b) + char_similarity(a, b)) / 2


# =========================
#  加载数据
# =========================

base_logits = load_jsonl("/root/model/Fun-Audio-Chat/eval_scripts/outputs_base_logits/predictions.jsonl")
base_by_id = {r['question_id']: r for r in base_logits}

# =========================
#  分析：按混淆类型分组，看 logit 差异
# =========================

print("=" * 90)
print("📊 交叉验证：Yes-Logit 分数 vs 文本相似度 (Base Model)")
print("=" * 90)

# 分组统计
categories = {
    'high_confusion': [],    # 高混淆 (sim > 0.5, rank=1)
    'moderate_confusion': [], # 中等混淆
    'low_similarity_confused': [], # 低相似但仍是最近项
    'tie_confusion': [],     # 并列混淆
    'non_confusion': [],     # 非混淆
}

for r in base_logits:
    if r.get('correct') or 'error' in r:
        continue

    gold = r.get('gold_answer', '')
    pred = r.get('predicted_answer', '')
    logit_options = r.get('yes_logits', [])

    if not logit_options:
        continue

    # 找 gold 和 pred 的 logit 值
    gold_logit = None
    pred_logit = None
    all_logits = []
    for opt in logit_options:
        all_logits.append(opt['yes_logit'])
        if opt['option_text'].strip() == gold.strip():
            gold_logit = opt['yes_logit']
        if opt['option_text'].strip() == pred.strip():
            pred_logit = opt['yes_logit']

    if gold_logit is None or pred_logit is None:
        # 模糊匹配
        for opt in logit_options:
            if char_similarity(opt['option_text'], gold) > 0.9 and gold_logit is None:
                gold_logit = opt['yes_logit']
            if char_similarity(opt['option_text'], pred) > 0.9 and pred_logit is None:
                pred_logit = opt['yes_logit']

    if gold_logit is None or pred_logit is None:
        continue

    logit_diff = pred_logit - gold_logit  # 正值表示模型更倾向错误选项
    logit_gap = abs(pred_logit - gold_logit)  # 绝对值差距
    max_other = max([l for l in all_logits if l != gold_logit and l != pred_logit], default=gold_logit)
    best_wrong_logit = max([l for l in all_logits if l != gold_logit], default=gold_logit)

    # 文本相似度
    sim = combined_similarity(pred, gold)

    # 混淆判断
    if sim > 0.5:
        confusion = 'high_confusion'
    elif sim > 0.3:
        confusion = 'moderate_confusion'
    elif logit_gap < 1.0:
        confusion = 'tie_confusion'
    elif logit_diff > 0 and sim > 0.15:
        confusion = 'low_similarity_confused'
    else:
        confusion = 'non_confusion'

    categories[confusion].append({
        'qid': r['question_id'],
        'question': r.get('question_text', '')[:80],
        'gold': gold[:60],
        'pred': pred[:60],
        'sim': round(sim, 4),
        'gold_logit': round(gold_logit, 4),
        'pred_logit': round(pred_logit, 4),
        'logit_diff': round(logit_diff, 4),
        'logit_gap': round(logit_gap, 4),
        'gold_rank': sum(1 for l in all_logits if l > gold_logit) + 1,
        'pred_rank': sum(1 for l in all_logits if l > pred_logit) + 1,
        'all_logits': sorted(all_logits, reverse=True),
    })


# =========================
#  打印分析
# =========================

# 表1: 各类混淆的 logit gap 统计
print("\n--- 各类错误类型的 Yes-Logit Gap 统计 ---")
print(f"{'类型':<30} {'数量':>5} {'平均Gap':>10} {'平均Sim':>10} {'Gap<0.5':>8} {'Gap<1.0':>8} {'Gold非Top2':>10}")
print("-" * 90)

for cat_name in ['high_confusion', 'moderate_confusion', 'low_similarity_confused', 'tie_confusion', 'non_confusion']:
    items = categories[cat_name]
    if not items:
        continue
    n = len(items)
    avg_gap = sum(x['logit_gap'] for x in items) / n
    avg_sim = sum(x['sim'] for x in items) / n
    gap_small = sum(1 for x in items if x['logit_gap'] < 0.5)
    gap_med = sum(1 for x in items if x['logit_gap'] < 1.0)
    gold_not_top2 = sum(1 for x in items if x['gold_rank'] > 2)

    print(f"{cat_name:<30} {n:>5} {avg_gap:>10.4f} {avg_sim:>10.4f} {gap_small:>8} {gap_med:>8} {gold_not_top2:>10}")


# 表2: 高混淆案例详情（带 logit 分数）
print(f"\n\n--- 高混淆错误详情 (sim>0.5) + Yes-Logit 分数 ---")
print(f"(共 {len(categories['high_confusion'])} 个案例，展示前15个)")
print()

for item in sorted(categories['high_confusion'], key=lambda x: -x['sim'])[:15]:
    print(f"Q: {item['question']}")
    print(f"  ✅ 正确答案: {item['gold']}")
    print(f"  ❌ 模型选了: {item['pred']}")
    print(f"  文本相似度: {item['sim']:.4f}")
    print(f"  Yes-Logit 分数:")
    print(f"    正确答案: {item['gold_logit']:.4f}  (排名: {item['gold_rank']})")
    print(f"    错误选项: {item['pred_logit']:.4f}  (排名: {item['pred_rank']})")
    print(f"    Logit Gap: {item['logit_gap']:.4f}  {'⚠️ 极小差距！' if item['logit_gap'] < 0.5 else ''}")
    print(f"    全部选项 logits: {item['all_logits']}")
    print()


# 表3: 非混淆案例也展示（对比）
print(f"\n--- 非混淆错误详情 (sim<0.15) + Yes-Logit 分数 (对比) ---")
print(f"(共 {len(categories['non_confusion'])} 个案例，展示前10个)")
print()

for item in sorted(categories['non_confusion'], key=lambda x: x['logit_gap'])[:10]:
    print(f"Q: {item['question']}")
    print(f"  ✅ 正确答案: {item['gold']}")
    print(f"  ❌ 模型选了: {item['pred']}")
    print(f"  文本相似度: {item['sim']:.4f}")
    print(f"  Yes-Logit 分数:")
    print(f"    正确答案: {item['gold_logit']:.4f}  (排名: {item['gold_rank']})")
    print(f"    错误选项: {item['pred_logit']:.4f}  (排名: {item['pred_rank']})")
    print(f"    Logit Gap: {item['logit_gap']:.4f}")
    print(f"    全部选项 logits: {item['all_logits']}")
    print()


# =========================
#  关键交叉验证指标
# =========================

print("=" * 90)
print("📊 关键交叉验证结论")
print("=" * 90)

# 1. 混淆型和 logit gap 的相关性
all_items = []
for cat_name, items in categories.items():
    all_items.extend(items)

# Spearman-like: 按 sim 分桶统计 logit gap
print("\n--- 文本相似度 vs Logit Gap (相似度越高，Gap应该越小) ---")
buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for lo, hi in buckets:
    bucket_items = [x for x in all_items if lo <= x['sim'] < hi]
    if not bucket_items:
        continue
    n = len(bucket_items)
    avg_gap = sum(x['logit_gap'] for x in bucket_items) / n
    avg_sim = sum(x['sim'] for x in bucket_items) / n
    gap_small = sum(1 for x in bucket_items if x['logit_gap'] < 0.5)
    gold_not_top1 = sum(1 for x in bucket_items if x['gold_rank'] > 1)
    print(f"  sim [{lo:.1f}-{hi:.1f}): n={n:3d}, avg_gap={avg_gap:.4f}, gap<0.5: {gap_small}/{n}, gold非第一: {gold_not_top1}/{n}")

# 2. 正确答案在 logit 中的排名分布
print("\n--- 正确答案的 Yes-Logit 排名分布 ---")
rank_dist = defaultdict(int)
for x in all_items:
    rank_dist[x['gold_rank']] += 1
for rank in sorted(rank_dist.keys()):
    print(f"  Rank {rank}: {rank_dist[rank]} ({rank_dist[rank]/len(all_items)*100:.1f}%)")

# 3. 有多少题目的 top-2 logits 之间差距极小（真正的混淆）
top2_gaps = []
for r in base_logits:
    if 'yes_logits' not in r:
        continue
    logits = sorted([opt['yes_logit'] for opt in r['yes_logits']], reverse=True)
    if len(logits) >= 2:
        top2_gaps.append(logits[0] - logits[1])

avg_top2_gap = sum(top2_gaps) / len(top2_gaps) if top2_gaps else 0
print(f"\n--- 全部 1607 题的 Top1-Top2 Logit Gap 统计 ---")
print(f"  平均 Gap: {avg_top2_gap:.4f}")
print(f"  Gap < 0.1: {sum(1 for g in top2_gaps if g < 0.1)} ({sum(1 for g in top2_gaps if g < 0.1)/len(top2_gaps)*100:.1f}%)")
print(f"  Gap < 0.5: {sum(1 for g in top2_gaps if g < 0.5)} ({sum(1 for g in top2_gaps if g < 0.5)/len(top2_gaps)*100:.1f}%)")
print(f"  Gap < 1.0: {sum(1 for g in top2_gaps if g < 1.0)} ({sum(1 for g in top2_gaps if g < 1.0)/len(top2_gaps)*100:.1f}%)")
print(f"  Gap >= 5.0: {sum(1 for g in top2_gaps if g >= 5.0)} ({sum(1 for g in top2_gaps if g >= 5.0)/len(top2_gaps)*100:.1f}%)  ← 模型很确定")
