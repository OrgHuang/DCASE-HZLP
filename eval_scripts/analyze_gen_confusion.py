#!/usr/bin/env python3
"""
Generate-based 标准评测的混淆分析。
对比 Base Model (无SFT) vs SFT Enhanced Model 在普通 Generate Eval 上的错误模式。
"""

import json
import re
import math
from collections import defaultdict, Counter
from difflib import SequenceMatcher


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def token_overlap(a, b):
    toks_a = set(re.findall(r'\w+', a.lower()))
    toks_b = set(re.findall(r'\w+', b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def char_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def combined_similarity(a, b):
    return (token_overlap(a, b) + char_similarity(a, b)) / 2


def analyze_errors(results, model_name):
    errors = [r for r in results if not r.get('correct') and 'error' not in r]
    all_q = [r for r in results if 'error' not in r]

    per_question = []
    confusion_count = 0
    non_confusion_count = 0

    for r in errors:
        gold = r.get('gold_answer', '')
        pred = r.get('prediction', '')
        choices = r.get('choices', [])

        if not choices or len(choices) < 2:
            continue

        # 找 gold 和 pred 的 index
        gold_idx = next((i for i, c in enumerate(choices) if c.strip() == gold.strip()), None)
        pred_idx = next((i for i, c in enumerate(choices) if c.strip() == pred.strip()), None)

        if gold_idx is None or pred_idx is None:
            # 模糊匹配
            for i, c in enumerate(choices):
                sim = char_similarity(c, gold)
                if sim > 0.85 and gold_idx is None:
                    gold_idx = i
                if char_similarity(c, pred) > 0.85 and pred_idx is None:
                    pred_idx = i

        if gold_idx is None or pred_idx is None:
            continue

        # 计算每个错误选项与 gold 的相似度
        wrong_sims = []
        for i, c in enumerate(choices):
            if i != gold_idx:
                wrong_sims.append({
                    'index': i, 'text': c,
                    'token_overlap': round(token_overlap(c, gold), 4),
                    'char_sim': round(char_similarity(c, gold), 4),
                    'combined': round(combined_similarity(c, gold), 4),
                })
        wrong_sims.sort(key=lambda x: -x['combined'])

        pred_wrong_sim = next((s for s in wrong_sims if s['index'] == pred_idx), None)
        if pred_wrong_sim is None:
            continue

        confusion_rank = next(i+1 for i, s in enumerate(wrong_sims) if s['index'] == pred_idx)
        top_sim = wrong_sims[0]['combined']
        pred_sim = pred_wrong_sim['combined']

        if confusion_rank == 1 and top_sim > 0.5:
            ctype = "high_confusion"
        elif confusion_rank == 1 and top_sim > 0.3:
            ctype = "moderate_confusion"
        elif confusion_rank == 1:
            ctype = "low_similarity_confused"
        elif confusion_rank == 2 and abs(top_sim - pred_sim) < 0.05:
            ctype = "tie_confusion"
        else:
            ctype = "non_confusion"

        if ctype != 'non_confusion':
            confusion_count += 1
        else:
            non_confusion_count += 1

        per_question.append({
            'question_id': r.get('question', r.get('question_id', '')),
            'question_text': r.get('question_text', ''),
            'gold_answer': gold,
            'predicted_answer': pred,
            'gold_index': gold_idx,
            'predicted_index': pred_idx,
            'confusion_type': ctype,
            'confusion_rank': confusion_rank,
            'pred_sim_to_gold': round(pred_sim, 4),
            'top_wrong_sim': round(top_sim, 4),
            'top_wrong_text': wrong_sims[0]['text'],
            'num_options': len(choices),
        })

    total = len(per_question)
    type_counts = Counter(r['confusion_type'] for r in per_question)
    avg_rank = sum(r['confusion_rank'] for r in per_question) / total if total else 0
    avg_sim = sum(r['pred_sim_to_gold'] for r in per_question) / total if total else 0
    avg_top_sim = sum(r['top_wrong_sim'] for r in per_question) / total if total else 0
    confusion_rate = confusion_count / total if total else 0

    # sim分桶
    sim_buckets = defaultdict(int)
    for r in per_question:
        s = r['pred_sim_to_gold']
        b = f"{int(s*5)/5:.1f}-{int(s*5)/5+0.2:.1f}"
        bucket = "0.0-0.2" if s < 0.2 else "0.2-0.4" if s < 0.4 else "0.4-0.6" if s < 0.6 else "0.6-0.8" if s < 0.8 else "0.8-1.0"
        sim_buckets[bucket] += 1

    stats = {
        'model': model_name,
        'total_samples': len(all_q),
        'total_errors': total,
        'accuracy': (len(all_q) - total) / len(all_q) if all_q else 0,
        'confusion_count': confusion_count,
        'non_confusion_count': non_confusion_count,
        'confusion_rate': confusion_rate,
        'type_breakdown': dict(type_counts),
        'avg_confusion_rank': round(avg_rank, 2),
        'avg_pred_sim_to_gold': round(avg_sim, 4),
        'avg_max_wrong_sim': round(avg_top_sim, 4),
        'sim_buckets': dict(sim_buckets),
    }
    return per_question, stats


def main():
    base_path = "/root/model/Fun-Audio-Chat/eval_scripts/outputs_base_gen/predictions.jsonl"
    enh_path = "/root/model/Fun-Audio-Chat/eval_scripts/outputs_enhanced/predictions.jsonl"

    base = load_jsonl(base_path)
    enh = load_jsonl(enh_path)
    print(f"Base: {len(base)} | Enhanced: {len(enh)}")

    base_errors, base_stats = analyze_errors(base, "Base Model (Generate)")
    enh_errors, enh_stats = analyze_errors(enh, "SFT Enhanced (Generate)")

    # ============ 打印 Base ============
    s = base_stats
    print(f"\n{'='*80}")
    print(f"📊 Base Model (Generate Eval, {s['accuracy']*100:.2f}%)")
    print(f"{'='*80}")
    print(f"  错误: {s['total_errors']} | 混淆型: {s['confusion_count']} ({s['confusion_rate']*100:.1f}%) | 非混淆型: {s['non_confusion_count']} ({(1-s['confusion_rate'])*100:.1f}%)")
    print(f"  平均混淆排名: {s['avg_confusion_rank']} | 平均sim: {s['avg_pred_sim_to_gold']:.4f} | 最像错选项平均sim: {s['avg_max_wrong_sim']:.4f}")
    print(f"  混淆类型: {s['type_breakdown']}")
    print(f"  相似度分桶: {s['sim_buckets']}")

    # 高混淆案例
    high = [r for r in base_errors if r['confusion_type'] == 'high_confusion']
    print(f"\n  --- 高混淆案例 (前10, 共{len(high)}) ---")
    for r in sorted(high, key=lambda x: -x['pred_sim_to_gold'])[:10]:
        print(f"\n  [{r['question_id']}] {r['question_text'][:100]}")
        print(f"  ✅ {r['gold_answer'][:80]}")
        print(f"  ❌ {r['predicted_answer'][:80]}")
        print(f"  sim={r['pred_sim_to_gold']:.4f} rank={r['confusion_rank']}/{r['num_options']-1}")

    # 非混淆案例
    non = [r for r in base_errors if r['confusion_type'] == 'non_confusion']
    print(f"\n  --- 非混淆案例 (前5, 共{len(non)}) ---")
    for r in sorted(non, key=lambda x: x['pred_sim_to_gold'])[:5]:
        print(f"\n  [{r['question_id']}] {r['question_text'][:100]}")
        print(f"  ✅ {r['gold_answer'][:80]}")
        print(f"  ❌ {r['predicted_answer'][:80]}")
        print(f"  sim={r['pred_sim_to_gold']:.4f} rank={r['confusion_rank']}/{r['num_options']-1}")

    # ============ 打印 Enhanced ============
    s = enh_stats
    print(f"\n{'='*80}")
    print(f"📊 SFT Enhanced Model (Generate Eval, {s['accuracy']*100:.2f}%)")
    print(f"{'='*80}")
    print(f"  错误: {s['total_errors']} | 混淆型: {s['confusion_count']} ({s['confusion_rate']*100:.1f}%) | 非混淆型: {s['non_confusion_count']} ({(1-s['confusion_rate'])*100:.1f}%)")
    print(f"  平均混淆排名: {s['avg_confusion_rank']} | 平均sim: {s['avg_pred_sim_to_gold']:.4f} | 最像错选项平均sim: {s['avg_max_wrong_sim']:.4f}")
    print(f"  混淆类型: {s['type_breakdown']}")
    print(f"  相似度分桶: {s['sim_buckets']}")

    high = [r for r in enh_errors if r['confusion_type'] == 'high_confusion']
    print(f"\n  --- 高混淆案例 (前10, 共{len(high)}) ---")
    for r in sorted(high, key=lambda x: -x['pred_sim_to_gold'])[:10]:
        print(f"\n  [{r['question_id']}] {r['question_text'][:100]}")
        print(f"  ✅ {r['gold_answer'][:80]}")
        print(f"  ❌ {r['predicted_answer'][:80]}")
        print(f"  sim={r['pred_sim_to_gold']:.4f} rank={r['confusion_rank']}/{r['num_options']-1}")

    non = [r for r in enh_errors if r['confusion_type'] == 'non_confusion']
    print(f"\n  --- 非混淆案例 (前5, 共{len(non)}) ---")
    for r in sorted(non, key=lambda x: x['pred_sim_to_gold'])[:5]:
        print(f"\n  [{r['question_id']}] {r['question_text'][:100]}")
        print(f"  ✅ {r['gold_answer'][:80]}")
        print(f"  ❌ {r['predicted_answer'][:80]}")
        print(f"  sim={r['pred_sim_to_gold']:.4f} rank={r['confusion_rank']}/{r['num_options']-1}")

    # ============ 对比 ============
    base_err_ids = {r['question_id'] for r in base_errors}
    enh_err_ids = {r['question_id'] for r in enh_errors}
    common = base_err_ids & enh_err_ids
    fixed = base_err_ids - enh_err_ids
    new = enh_err_ids - base_err_ids

    same_wrong = sum(1 for qid in common
                     if next(r['predicted_answer'] for r in base_errors if r['question_id'] == qid).strip()
                     == next(r['predicted_answer'] for r in enh_errors if r['question_id'] == qid).strip())

    print(f"\n{'='*80}")
    print(f"📊 Base vs Enhanced 对比")
    print(f"{'='*80}")
    print(f"  Base 错误:     {base_stats['total_errors']} (准确率 {base_stats['accuracy']*100:.2f}%)")
    print(f"  Enhanced 错误:  {enh_stats['total_errors']} (准确率 {enh_stats['accuracy']*100:.2f}%)")
    print(f"  Enhanced 修复: {len(fixed)} | 新增错误: {len(new)} | 共同错误: {len(common)}")
    print(f"  共同错误中选同一错答案: {same_wrong} ({same_wrong/max(1,len(common))*100:.1f}%)")
    print(f"  共同错误中选不同答案:   {len(common)-same_wrong}")

    print(f"\n  混淆率:  Base {base_stats['confusion_rate']*100:.1f}% → Enhanced {enh_stats['confusion_rate']*100:.1f}%")
    print(f"  平均sim: Base {base_stats['avg_pred_sim_to_gold']:.4f} → Enhanced {enh_stats['avg_pred_sim_to_gold']:.4f}")

    # 修复的错误中混淆型占比
    fixed_conf = sum(1 for r in base_errors if r['question_id'] in fixed and r['confusion_type'] != 'non_confusion')
    fixed_non = sum(1 for r in base_errors if r['question_id'] in fixed and r['confusion_type'] == 'non_confusion')
    print(f"\n  Enhanced 修复的错误:")
    print(f"    混淆型: {fixed_conf}, 非混淆型: {fixed_non}")

    new_conf = sum(1 for r in enh_errors if r['question_id'] in new and r['confusion_type'] != 'non_confusion')
    new_non = sum(1 for r in enh_errors if r['question_id'] in new and r['confusion_type'] == 'non_confusion')
    print(f"  Enhanced 新增的错误:")
    print(f"    混淆型: {new_conf}, 非混淆型: {new_non}")

    # 保存
    output = {
        'base_stats': base_stats,
        'enhanced_stats': enh_stats,
        'comparison': {
            'base_errors': base_stats['total_errors'],
            'enh_errors': enh_stats['total_errors'],
            'fixed': len(fixed),
            'new': len(new),
            'common': len(common),
            'same_wrong': same_wrong,
        },
        'base_error_details': base_errors,
        'enhanced_error_details': enh_errors,
    }
    out_path = "/root/model/Fun-Audio-Chat/eval_scripts/outputs_compare/gen_confusion_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
