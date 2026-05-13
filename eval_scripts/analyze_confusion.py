#!/usr/bin/env python3
"""
Confusion Analysis: 分析多选题错误是否因为混淆选项导致。

核心思路：
  对每道错题，计算模型选中的选项与正确答案的文本相似度，
  并与其他错误选项与正确答案的相似度对比。
  如果模型选中的是"和正确答案最像的错误选项"，则判定为"混淆型错误"。

对比维度：Base Model vs SFT Enhanced Model
"""

import json
import sys
import re
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def token_overlap(a, b):
    """Jaccard similarity on word-level tokens."""
    toks_a = set(re.findall(r'\w+', a.lower()))
    toks_b = set(re.findall(r'\w+', b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def char_similarity(a, b):
    """Character-level sequence similarity (longest common subsequence ratio)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def combined_similarity(a, b):
    """Average of token overlap and char similarity."""
    return (token_overlap(a, b) + char_similarity(a, b)) / 2


def analyze_model_errors(results, model_name):
    """
    对每个模型的错误进行混淆分析。

    返回:
      - per_question: 每题详细分析
      - confusion_stats: 混淆统计
      - category_breakdown: 按混淆程度分类
    """
    errors = [r for r in results if not r.get('correct') and 'error' not in r]
    all_questions = [r for r in results if 'error' not in r]

    per_question = []
    confusion_count = 0
    random_count = 0
    tie_count = 0

    for r in errors:
        gold = r.get('gold_answer', '')
        pred = r.get('predicted_answer', r.get('prediction', ''))
        choices = r.get('choices', [])

        # 如果是从 logits 格式来的，需要从 yes_logits 提取 options
        if not choices and 'yes_logits' in r:
            choices = [opt['option_text'] for opt in r['yes_logits']]
        if not choices and 'options' in r:
            choices = [opt.get('option_text', opt.get('text', '')) for opt in r.get('options', [])]

        if not choices or len(choices) < 2:
            continue

        # 找到 gold 和 pred 在 choices 中的索引
        gold_idx = None
        pred_idx = None
        for i, c in enumerate(choices):
            if c.strip() == gold.strip():
                gold_idx = i
            if c.strip() == pred.strip():
                pred_idx = i

        if gold_idx is None or pred_idx is None:
            # 尝试模糊匹配
            for i, c in enumerate(choices):
                if char_similarity(c, gold) > 0.9:
                    gold_idx = i
                if char_similarity(c, pred) > 0.9:
                    pred_idx = i

        # 计算每个选项与正确答案的相似度
        sim_to_gold = []
        for i, c in enumerate(choices):
            if i != gold_idx:
                sim_to_gold.append({
                    'index': i,
                    'text': c,
                    'token_overlap': round(token_overlap(c, gold), 4),
                    'char_sim': round(char_similarity(c, gold), 4),
                    'combined': round(combined_similarity(c, gold), 4),
                })

        sim_to_gold.sort(key=lambda x: x['combined'], reverse=True)

        # 判断：模型选的选项是不是所有错误选项中最像正确答案的？
        pred_sim = None
        for s in sim_to_gold:
            if s['index'] == pred_idx:
                pred_sim = s
                break

        if pred_sim is None:
            continue

        # 混淆排名 (1 = 最像正确答案的错误选项)
        confusion_rank = next((i+1 for i, s in enumerate(sim_to_gold) if s['index'] == pred_idx), len(sim_to_gold))

        # 判定混淆类型
        top_sim = sim_to_gold[0]['combined']

        if confusion_rank == 1:
            if top_sim > 0.5:
                confusion_type = "high_confusion"  # 明显被混淆
            elif top_sim > 0.3:
                confusion_type = "moderate_confusion"  # 中等混淆
            else:
                confusion_type = "low_similarity_confused"  # 低相似但仍选了最近似项
            confusion_count += 1
        elif confusion_rank == 2 and abs(sim_to_gold[0]['combined'] - pred_sim['combined']) < 0.05:
            confusion_type = "tie_confusion"  # 几乎并列
            confusion_count += 1
        else:
            confusion_type = "non_confusion"  # 非混淆型错误
            random_count += 1

        qid = r.get('question_id', r.get('question', ''))
        per_question.append({
            'question_id': qid,
            'question_text': r.get('question_text', ''),
            'gold_answer': gold,
            'predicted_answer': pred,
            'gold_index': gold_idx,
            'predicted_index': pred_idx,
            'confusion_rank': confusion_rank,
            'confusion_type': confusion_type,
            'pred_sim_to_gold': pred_sim['combined'],
            'top_wrong_sim_to_gold': sim_to_gold[0]['combined'],
            'top_wrong_text': sim_to_gold[0]['text'],
            'all_wrong_sims': sim_to_gold,
            'num_options': len(choices),
        })

    total_errors = len(per_question)

    # 分类统计
    type_counts = Counter(r['confusion_type'] for r in per_question)

    # 按混淆相似度分桶
    sim_buckets = defaultdict(int)
    for r in per_question:
        sim = r['pred_sim_to_gold']
        bucket = f"0.0-0.2" if sim < 0.2 else f"0.2-0.4" if sim < 0.4 else f"0.4-0.6" if sim < 0.6 else f"0.6-0.8" if sim < 0.8 else f"0.8-1.0"
        sim_buckets[bucket] += 1

    # 平均混淆指标
    avg_confusion_rank = sum(r['confusion_rank'] for r in per_question) / total_errors if total_errors else 0
    avg_pred_sim = sum(r['pred_sim_to_gold'] for r in per_question) / total_errors if total_errors else 0
    avg_top_wrong_sim = sum(r['top_wrong_sim_to_gold'] for r in per_question) / total_errors if total_errors else 0
    confusion_rate = confusion_count / total_errors if total_errors else 0

    # 每道题的"混淆潜能"（所有错误选项中的最高相似度）
    avg_max_wrong_sim = avg_top_wrong_sim

    stats = {
        'model': model_name,
        'total_samples': len(all_questions),
        'total_errors': total_errors,
        'accuracy': (len(all_questions) - total_errors) / len(all_questions) if all_questions else 0,
        'confusion_count': confusion_count,
        'non_confusion_count': random_count,
        'confusion_rate': confusion_rate,
        'type_breakdown': dict(type_counts),
        'avg_confusion_rank': round(avg_confusion_rank, 2),
        'avg_pred_similarity_to_gold': round(avg_pred_sim, 4),
        'avg_max_wrong_similarity_to_gold': round(avg_max_wrong_sim, 4),
        'similarity_buckets': dict(sim_buckets),
    }

    return per_question, stats


def compare_models(base_errors, base_stats, enh_errors, enh_stats):
    """对比 Base 和 Enhanced 模型的错误模式。"""

    # 找出两个模型都错的题目（共同错误）
    base_error_ids = {r['question_id'] for r in base_errors}
    enh_error_ids = {r['question_id'] for r in enh_errors}
    common_errors = base_error_ids & enh_error_ids
    base_only_errors = base_error_ids - enh_error_ids
    enh_only_errors = enh_error_ids - base_error_ids

    # 共同错误中，两个模型选同一错误选项的（系统性混淆）
    same_wrong = 0
    diff_wrong = 0
    for qid in common_errors:
        base_pred = next(r['predicted_answer'] for r in base_errors if r['question_id'] == qid)
        enh_pred = next(r['predicted_answer'] for r in enh_errors if r['question_id'] == qid)
        if base_pred.strip() == enh_pred.strip():
            same_wrong += 1
        else:
            diff_wrong += 1

    comparison = {
        'common_errors': len(common_errors),
        'base_only_errors': len(base_only_errors),
        'enh_only_errors': len(enh_only_errors),
        'same_wrong_answer': same_wrong,
        'different_wrong_answer': diff_wrong,
        'enh_fixed_from_base': len(base_only_errors),  # Enhanced 修复了这些
        'enh_new_errors': len(enh_only_errors),  # Enhanced 新增的错误

        # 混淆指标对比
        'base_confusion_rate': base_stats['confusion_rate'],
        'enh_confusion_rate': enh_stats['confusion_rate'],
        'base_avg_sim_to_gold': base_stats['avg_pred_similarity_to_gold'],
        'enh_avg_sim_to_gold': enh_stats['avg_pred_similarity_to_gold'],
    }

    return comparison


def print_analysis(model_name, stats, errors, top_n=10):
    """打印详细分析报告。"""
    print(f"\n{'='*80}")
    print(f"📊 {model_name}")
    print(f"{'='*80}")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  正确数:   {stats['total_samples'] - stats['total_errors']}")
    print(f"  错误数:   {stats['total_errors']}")
    print(f"  准确率:   {stats['accuracy']*100:.2f}%")
    print()
    print(f"  混淆型错误 ({stats['confusion_count']}): {stats['confusion_rate']*100:.1f}%")
    print(f"  非混淆型错误 ({stats['non_confusion_count']}): {(1-stats['confusion_rate'])*100:.1f}%")
    print(f"  平均混淆排名: {stats['avg_confusion_rank']} (1=选了最像的错选项)")
    print(f"  预测-正确答案平均相似度: {stats['avg_pred_similarity_to_gold']:.4f}")
    print(f"  最像错选项-正确答案平均相似度: {stats['avg_max_wrong_similarity_to_gold']:.4f}")
    print()
    print(f"  混淆类型分布:")
    for t, c in sorted(stats['type_breakdown'].items(), key=lambda x: -x[1]):
        print(f"    {t}: {c} ({c/stats['total_errors']*100:.1f}%)")
    print()
    print(f"  相似度分桶 (预测vs正确):")
    for b in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]:
        c = stats['similarity_buckets'].get(b, 0)
        bar = "█" * (c * 5 // stats['total_errors'] + 1) if stats['total_errors'] > 0 else ""
        print(f"    {b}: {c:4d} {bar}")

    # 展示典型混淆案例
    high_conf = [r for r in errors if r['confusion_type'] == 'high_confusion']
    print(f"\n  --- 高混淆错误案例 (前{min(top_n, len(high_conf))}个) ---")
    for r in sorted(high_conf, key=lambda x: -x['pred_sim_to_gold'])[:top_n]:
        print(f"\n  Q: {r['question_text'][:100]}")
        print(f"  ✅ 正确答案: {r['gold_answer'][:80]}")
        print(f"  ❌ 模型选了: {r['predicted_answer'][:80]}")
        print(f"  混淆排名: {r['confusion_rank']}/{r['num_options']-1}, 相似度: {r['pred_sim_to_gold']:.4f}")
        print(f"  最像的错误选项: {r['top_wrong_text'][:80]} (sim={r['top_wrong_sim_to_gold']:.4f})")

    # 展示非混淆案例
    non_conf = [r for r in errors if r['confusion_type'] == 'non_confusion']
    print(f"\n  --- 非混淆错误案例 (前{min(5, len(non_conf))}个) ---")
    for r in sorted(non_conf, key=lambda x: x['pred_sim_to_gold'])[:5]:
        print(f"\n  Q: {r['question_text'][:100]}")
        print(f"  ✅ 正确答案: {r['gold_answer'][:80]}")
        print(f"  ❌ 模型选了: {r['predicted_answer'][:80]}")
        print(f"  混淆排名: {r['confusion_rank']}/{r['num_options']-1}, 相似度: {r['pred_sim_to_gold']:.4f}")


def main():
    # 加载数据
    base_path = "/root/model/Fun-Audio-Chat/eval_scripts/outputs_base_logits/predictions.jsonl"
    enh_path = "/root/model/Fun-Audio-Chat/eval_scripts/outputs_enhanced/predictions.jsonl"

    base_results = load_jsonl(base_path)
    enh_results = load_jsonl(enh_path)

    print(f"Base model: {len(base_results)} results")
    print(f"Enhanced model: {len(enh_results)} results")

    # 分析
    base_errors, base_stats = analyze_model_errors(base_results, "Base Model (Logits Eval)")
    enh_errors, enh_stats = analyze_model_errors(enh_results, "SFT Enhanced Model (Generate Eval)")

    # 打印
    print_analysis("Base Model (Logits Eval, 60.61%)", base_stats, base_errors)
    print_analysis("SFT Enhanced Model (Generate Eval, 64.28%)", enh_stats, enh_errors)

    # 对比
    comp = compare_models(base_errors, base_stats, enh_errors, enh_stats)

    print(f"\n{'='*80}")
    print(f"📊 Base vs Enhanced 对比分析")
    print(f"{'='*80}")
    print(f"  Base 错误:    {base_stats['total_errors']} (准确率 {base_stats['accuracy']*100:.2f}%)")
    print(f"  Enhanced 错误: {enh_stats['total_errors']} (准确率 {enh_stats['accuracy']*100:.2f}%)")
    print(f"  Enhanced 修复了 Base 的: {comp['enh_fixed_from_base']} 题")
    print(f"  Enhanced 新增错误:        {comp['enh_new_errors']} 题")
    print(f"  共同错误:                  {comp['common_errors']} 题")
    print(f"  共同错误中选同一错答案:    {comp['same_wrong_answer']} 题")
    print(f"  共同错误中选不同错答案:    {comp['different_wrong_answer']} 题")
    print()
    print(f"  混淆率对比:")
    print(f"    Base:     {comp['base_confusion_rate']*100:.1f}% (错误中混淆型占比)")
    print(f"    Enhanced: {comp['enh_confusion_rate']*100:.1f}% (错误中混淆型占比)")
    print(f"  错误答案与正确答案平均相似度:")
    print(f"    Base:     {comp['base_avg_sim_to_gold']:.4f}")
    print(f"    Enhanced: {comp['enh_avg_sim_to_gold']:.4f}")

    # 分析 Enhanced 修复的问题类型
    print(f"\n  --- Enhanced 修复的 Base 错误 (抽样) ---")
    fixed_errors = [r for r in base_errors if r['question_id'] in comp and r['question_id'] not in {e['question_id'] for e in enh_errors}]
    # Actually let me recompute
    base_error_ids = {r['question_id'] for r in base_errors}
    enh_error_ids = {r['question_id'] for r in enh_errors}
    fixed_ids = base_error_ids - enh_error_ids

    # 分析修复的错误中，原本是混淆型还是非混淆型
    fixed_confusion = sum(1 for r in base_errors if r['question_id'] in fixed_ids and r['confusion_type'] in ('high_confusion', 'moderate_confusion', 'low_similarity_confused', 'tie_confusion'))
    fixed_non_confusion = sum(1 for r in base_errors if r['question_id'] in fixed_ids and r['confusion_type'] == 'non_confusion')
    print(f"  修复的错误中 - 混淆型: {fixed_confusion}, 非混淆型: {fixed_non_confusion}")

    # 分析 Enhanced 新增的错误
    new_ids = enh_error_ids - base_error_ids
    new_confusion = sum(1 for r in enh_errors if r['question_id'] in new_ids and r['confusion_type'] in ('high_confusion', 'moderate_confusion', 'low_similarity_confused', 'tie_confusion'))
    new_non_confusion = sum(1 for r in enh_errors if r['question_id'] in new_ids and r['confusion_type'] == 'non_confusion')
    print(f"  新增的错误中 - 混淆型: {new_confusion}, 非混淆型: {new_non_confusion}")

    # 展示新增混淆型错误案例
    print(f"\n  --- Enhanced 新增混淆型错误案例 ---")
    new_conf_errors = [r for r in enh_errors if r['question_id'] in new_ids and r['confusion_type'] == 'high_confusion']
    for r in sorted(new_conf_errors, key=lambda x: -x['pred_sim_to_gold'])[:5]:
        print(f"\n  Q: {r['question_text'][:100]}")
        print(f"  ✅ 正确答案: {r['gold_answer'][:80]}")
        print(f"  ❌ 模型选了: {r['predicted_answer'][:80]}")
        print(f"  相似度: {r['pred_sim_to_gold']:.4f}")

    # 保存详细结果
    output = {
        'base_stats': base_stats,
        'enhanced_stats': enh_stats,
        'comparison': comp,
        'base_error_details': base_errors,
        'enhanced_error_details': enh_errors,
    }
    out_path = "/root/model/Fun-Audio-Chat/eval_scripts/outputs_compare/confusion_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细分析结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
