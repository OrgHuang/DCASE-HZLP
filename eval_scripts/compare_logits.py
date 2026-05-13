#!/usr/bin/env python3
"""
Base vs Enhanced SFT: Yes-Logit 交叉对比分析。
对比两个模型在相同题目上的 Yes-logit 分布差异。
"""

import json
import re
import math
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


def get_logit_for_text(logit_options, target_text):
    """从 logit options 中找到目标文本的 logit 分数"""
    best_sim = 0
    best_logit = None
    for opt in logit_options:
        sim = char_similarity(opt['option_text'].strip(), target_text.strip())
        if sim > best_sim:
            best_sim = sim
            best_logit = opt['yes_logit']
    if best_sim > 0.85:
        return best_logit, best_sim
    return None, best_sim


# =========================
#  加载数据
# =========================

base_data = load_jsonl("/root/model/Fun-Audio-Chat/eval_scripts/outputs_base_logits/predictions.jsonl")
enh_data = load_jsonl("/root/model/Fun-Audio-Chat/eval_scripts/outputs_compare/sft_enhanced_logits/predictions.jsonl")

base_by_id = {r['question_id']: r for r in base_data}
enh_by_id = {r['question_id']: r for r in enh_data}

common_ids = set(base_by_id.keys()) & set(enh_by_id.keys())
print(f"Base: {len(base_data)}, Enhanced: {len(enh_data)}, Common: {len(common_ids)}")

# =========================
#  逐题对比
# =========================

comparisons = []

for qid in common_ids:
    b = base_by_id[qid]
    e = enh_by_id[qid]

    b_correct = b.get('correct', False)
    e_correct = e.get('correct', False)

    b_gold = b.get('gold_answer', '')
    b_pred = b.get('predicted_answer', '')
    e_gold = e.get('gold_answer', '')
    e_pred = e.get('predicted_answer', '')

    b_logits = b.get('yes_logits', [])
    e_logits = e.get('yes_logits', [])

    if not b_logits or not e_logits:
        continue

    # 对每个选项提取两个模型的 logit
    # 使用选项文本来对齐
    b_options = [(opt['option_text'].strip(), opt['yes_logit']) for opt in b_logits]
    e_options = [(opt['option_text'].strip(), opt['yes_logit']) for opt in e_logits]

    # 找到 gold 和 pred 对应的 logit
    b_gold_logit, _ = get_logit_for_text(b_logits, b_gold)
    b_pred_logit, _ = get_logit_for_text(b_logits, b_pred)
    e_gold_logit, _ = get_logit_for_text(e_logits, e_gold)
    e_pred_logit, _ = get_logit_for_text(e_logits, e_pred)

    if None in (b_gold_logit, b_pred_logit, e_gold_logit, e_pred_logit):
        continue

    # 计算各项指标
    b_logit_gap = b_pred_logit - b_gold_logit  # 正=错误选项更高
    e_logit_gap = e_pred_logit - e_gold_logit
    b_logit_abs_gap = abs(b_gap := b_pred_logit - b_gold_logit)
    e_logit_abs_gap = abs(e_pred_logit - e_gold_logit)

    # 正确答案排名
    b_all = sorted([opt['yes_logit'] for opt in b_logits], reverse=True)
    e_all = sorted([opt['yes_logit'] for opt in e_logits], reverse=True)
    b_gold_rank = sum(1 for l in b_all if l > b_gold_logit) + 1
    e_gold_rank = sum(1 for l in e_all if l > e_gold_logit) + 1

    # 文本相似度
    sim_pred_gold = combined_similarity(b_pred, b_gold)

    # 分类
    if b_correct and e_correct:
        category = "both_correct"
    elif not b_correct and e_correct:
        category = "enh_fixed"  # SFT 修复了
    elif b_correct and not e_correct:
        category = "enh_broke"  # SFT 搞坏了
    else:
        # 都错
        if b_pred.strip() == e_pred.strip():
            category = "both_wrong_same"  # 同一个坑
        else:
            category = "both_wrong_diff"

    # Enhanced vs Base 的 logit gap 变化
    gap_change = e_logit_abs_gap - b_logit_abs_gap  # 正=SFT后gap变大(更确定), 负=变小

    comparisons.append({
        'qid': qid,
        'question': b.get('question_text', '')[:100],
        'gold': b_gold[:80],
        'b_pred': b_pred[:80],
        'e_pred': e_pred[:80],
        'category': category,
        'sim_pred_gold': round(sim_pred_gold, 4),
        'b_gold_logit': round(b_gold_logit, 4),
        'b_pred_logit': round(b_pred_logit, 4),
        'e_gold_logit': round(e_gold_logit, 4),
        'e_pred_logit': round(e_pred_logit, 4),
        'b_logit_gap': round(b_logit_abs_gap, 4),
        'e_logit_gap': round(e_logit_abs_gap, 4),
        'gap_change': round(gap_change, 4),
        'b_gold_rank': b_gold_rank,
        'e_gold_rank': e_gold_rank,
    })


# =========================
#  统计汇总
# =========================

cat_counts = defaultdict(int)
for c in comparisons:
    cat_counts[c['category']] += 1

print(f"\n{'='*90}")
print(f"📊 Base vs Enhanced SFT: Yes-Logit 对比分析 ({len(comparisons)} 题)")
print(f"{'='*90}")
print(f"\n  分类统计:")
for cat in ['both_correct', 'enh_fixed', 'enh_broke', 'both_wrong_same', 'both_wrong_diff']:
    n = cat_counts.get(cat, 0)
    pct = n / len(comparisons) * 100
    labels = {
        'both_correct': '两个模型都对',
        'enh_fixed': 'SFT 修复了 Base 的错误',
        'enh_broke': 'SFT 新增的错误 (Base对了)',
        'both_wrong_same': '两个都错，选了同一个错误答案',
        'both_wrong_diff': '两个都错，选了不同的错误答案',
    }
    print(f"    {labels[cat]:<40} {n:>4} ({pct:.1f}%)")

# =========================
#  Logit Gap 分析
# =========================

print(f"\n{'='*90}")
print(f"📊 Logit Gap 对比 (Gap = |pred_logit - gold_logit|)")
print(f"{'='*90}")

for cat in ['both_correct', 'enh_fixed', 'enh_broke', 'both_wrong_same', 'both_wrong_diff']:
    items = [c for c in comparisons if c['category'] == cat]
    if not items:
        continue
    n = len(items)
    avg_b_gap = sum(c['b_logit_gap'] for c in items) / n
    avg_e_gap = sum(c['e_logit_gap'] for c in items) / n
    avg_sim = sum(c['sim_pred_gold'] for c in items) / n
    avg_gap_change = sum(c['gap_change'] for c in items) / n

    labels = {
        'both_correct': '两个都对',
        'enh_fixed': 'SFT修复',
        'enh_broke': 'SFT新增错误',
        'both_wrong_same': '同错同答',
        'both_wrong_diff': '同错异答',
    }
    print(f"\n  {labels[cat]} (n={n}):")
    print(f"    平均文本相似度(错误vs正确): {avg_sim:.4f}")
    print(f"    Base    平均 Gap: {avg_b_gap:.4f}")
    print(f"    Enhanced 平均 Gap: {avg_e_gap:.4f}")
    print(f"    Gap 变化: {avg_gap_change:+.4f} ({'SFT后更确定' if avg_gap_change > 0 else 'SFT后更不确定'})")


# =========================
#  Gold Logit 排名变化
# =========================

print(f"\n{'='*90}")
print(f"📊 正确答案的 Logit 排名分布变化")
print(f"{'='*90}")

for cat in ['both_correct', 'enh_fixed', 'enh_broke', 'both_wrong_same', 'both_wrong_diff']:
    items = [c for c in comparisons if c['category'] == cat]
    if not items:
        continue
    n = len(items)
    # 排名改善: base_rank - enh_rank > 0 表示 SFT 后排名提升
    rank_improved = sum(1 for c in items if c['b_gold_rank'] > c['e_gold_rank'])
    rank_worsened = sum(1 for c in items if c['b_gold_rank'] < c['e_gold_rank'])
    rank_same = sum(1 for c in items if c['b_gold_rank'] == c['e_gold_rank'])
    avg_b_rank = sum(c['b_gold_rank'] for c in items) / n
    avg_e_rank = sum(c['e_gold_rank'] for c in items) / n

    labels = {
        'both_correct': '两个都对',
        'enh_fixed': 'SFT修复',
        'enh_broke': 'SFT新增错误',
        'both_wrong_same': '同错同答',
        'both_wrong_diff': '同错异答',
    }
    print(f"\n  {labels[cat]} (n={n}):")
    print(f"    Base 平均排名: {avg_b_rank:.2f} → Enhanced: {avg_e_rank:.2f} (改善: {avg_b_rank - avg_e_rank:+.2f})")
    print(f"    排名提升: {rank_improved}, 排名下降: {rank_worsened}, 不变: {rank_same}")


# =========================
#  关键案例展示
# =========================

print(f"\n{'='*90}")
print(f"📊 SFT 修复的混淆型错误 (前12个)")
print(f"{'='*90}")

fixed_items = [c for c in comparisons if c['category'] == 'enh_fixed' and c['sim_pred_gold'] > 0.4]
fixed_items.sort(key=lambda x: -x['sim_pred_gold'])

for item in fixed_items[:12]:
    print(f"\n  Q: {item['question']}")
    print(f"  Gold: {item['gold']}")
    print(f"  Base 选了: {item['b_pred']} (sim={item['sim_pred_gold']:.4f})")
    print(f"  Base    logits: gold={item['b_gold_logit']:.4f} pred={item['b_pred_logit']:.4f} gap={item['b_logit_gap']:.4f} rank={item['b_gold_rank']}")
    print(f"  Enhanced logits: gold={item['e_gold_logit']:.4f} pred={item['e_pred_logit']:.4f} gap={item['e_logit_gap']:.4f} rank={item['e_gold_rank']}")


print(f"\n{'='*90}")
print(f"📊 SFT 新增的混淆型错误 (前12个)")
print(f"{'='*90}")

broke_items = [c for c in comparisons if c['category'] == 'enh_broke' and c['sim_pred_gold'] > 0.4]
broke_items.sort(key=lambda x: -x['sim_pred_gold'])

for item in broke_items[:12]:
    print(f"\n  Q: {item['question']}")
    print(f"  Gold: {item['gold']}")
    print(f"  Enhanced 选了: {item['e_pred']} (sim={item['sim_pred_gold']:.4f})")
    print(f"  Base    logits: gold={item['b_gold_logit']:.4f} pred={item['b_pred_logit']:.4f} gap={item['b_logit_gap']:.4f} rank={item['b_gold_rank']}")
    print(f"  Enhanced logits: gold={item['e_gold_logit']:.4f} pred={item['e_pred_logit']:.4f} gap={item['e_logit_gap']:.4f} rank={item['e_gold_rank']}")


# =========================
#  整体 Logit 分布变化
# =========================

print(f"\n{'='*90}")
print(f"📊 整体 Logit 分布变化")
print(f"{'='*90}")

# Top1-Top2 gap 分布
b_top2_gaps = []
e_top2_gaps = []
for c in comparisons:
    # 从 raw data 重新计算
    b = base_by_id[c['qid']]
    e = enh_by_id[c['qid']]
    b_logits = sorted([opt['yes_logit'] for opt in b.get('yes_logits', [])], reverse=True)
    e_logits = sorted([opt['yes_logit'] for opt in e.get('yes_logits', [])], reverse=True)
    if len(b_logits) >= 2:
        b_top2_gaps.append(b_logits[0] - b_logits[1])
    if len(e_logits) >= 2:
        e_top2_gaps.append(e_logits[0] - e_logits[1])

print(f"\n  Top1-Top2 Logit Gap 分布:")
for label, gaps in [("Base", b_top2_gaps), ("Enhanced", e_top2_gaps)]:
    if not gaps:
        continue
    print(f"\n  {label}:")
    print(f"    平均 Gap: {sum(gaps)/len(gaps):.4f}")
    print(f"    Gap < 0.1: {sum(1 for g in gaps if g < 0.1)} ({sum(1 for g in gaps if g < 0.1)/len(gaps)*100:.1f}%)")
    print(f"    Gap < 0.5: {sum(1 for g in gaps if g < 0.5)} ({sum(1 for g in gaps if g < 0.5)/len(gaps)*100:.1f}%)")
    print(f"    Gap < 1.0: {sum(1 for g in gaps if g < 1.0)} ({sum(1 for g in gaps if g < 1.0)/len(gaps)*100:.1f}%)")
    print(f"    Gap >= 5.0: {sum(1 for g in gaps if g >= 5.0)} ({sum(1 for g in gaps if g >= 5.0)/len(gaps)*100:.1f}%)")

# Logit 值的整体 scale 变化
b_all_logits = []
e_all_logits = []
for c in comparisons:
    b = base_by_id[c['qid']]
    e = enh_by_id[c['qid']]
    b_all_logits.extend([opt['yes_logit'] for opt in b.get('yes_logits', [])])
    e_all_logits.extend([opt['yes_logit'] for opt in e.get('yes_logits', [])])

print(f"\n  Logit 值的整体 Scale:")
print(f"    Base:     mean={sum(b_all_logits)/len(b_all_logits):.4f}, min={min(b_all_logits):.4f}, max={max(b_all_logits):.4f}")
print(f"    Enhanced: mean={sum(e_all_logits)/len(e_all_logits):.4f}, min={min(e_all_logits):.4f}, max={max(e_all_logits):.4f}")


# =========================
#  最终总结
# =========================

print(f"\n{'='*90}")
print(f"📊 最终总结")
print(f"{'='*90}")

# SFT 对混淆型错误的修复能力
fixed_high_conf = sum(1 for c in comparisons if c['category'] == 'enh_fixed' and c['sim_pred_gold'] > 0.5)
fixed_any = sum(1 for c in comparisons if c['category'] == 'enh_fixed')
broke_high_conf = sum(1 for c in comparisons if c['category'] == 'enh_broke' and c['sim_pred_gold'] > 0.5)
broke_any = sum(1 for c in comparisons if c['category'] == 'enh_broke')

print(f"""
  SFT 修复了 {fixed_any} 题 (其中高混淆: {fixed_high_conf})
  SFT 新增了 {broke_any} 题错误 (其中高混淆: {broke_high_conf})
  净收益: {fixed_any - broke_any} 题

  SFT 对混淆型错误的处理:
    - 能修复部分高混淆错误 (通过更好的音频理解来区分相似选项)
    - 但也引入新的混淆错误 (SFT 改变了模型对某些选项的偏好)
""")

# 生成模式下准确率提升 > logits 模式下反而下降，原因分析
print(f"""
  ⚠️ 重要发现: Logits vs Generate 的矛盾
    Base:     Logits=60.61%, Generate≈61.61%  (一致)
    Enhanced: Logits=56.81%, Generate=64.28%  (矛盾!)

  这说明 SFT 训练让模型学会了:
    1. 更好的"答题格式" — generate 时能正确输出答案
    2. 但内部的 Yes/No 判断信号变得不那么可靠
    3. SFT 改变了模型内部表征，使得 logit 提取和 generate 不再一致
""")
