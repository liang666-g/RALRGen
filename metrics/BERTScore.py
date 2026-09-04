"""
使用 BERTScore 对比评测模型的生成回复
运行示例：
python BERTScore.py --ref-file ./data/test.txt --core-file ./data/evaluation/llm_deepseek-v4-flash_com_5_log_5_20260514_23_47_52/predictions.txt
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from configuration import llm_config
import argparse
from bert_score import score


def load_texts(filepath, is_ref=False):
    """
    加载文本文件。
    """
    texts = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if is_ref:
                # 假设 test.txt 中第 6 个字段 (索引5) 是目标回复
                terms = line.split("-[split]-")
                if len(terms) >= 8:
                    texts.append(terms[5].strip())
            else:
                # 假设预测文件每一行要么是纯文本，要么是 src-[split]-tgt 格式
                parts = line.split("-[split]-", 1)
                text = parts[1].strip() if len(parts) == 2 else line.strip()
                texts.append(text)
                
    safe_texts = [t if t else "." for t in texts]
    return safe_texts


def compute_bertscore(preds, refs, model_type=None, verbose=False):
    if model_type is None:
        model_type = getattr(llm_config, "eval_model", "roberta-large")
        
    # 防止空字符串导致 bert_score 报错
    safe_preds = [p if p and str(p).strip() else "." for p in preds]
    safe_refs = [r if r and str(r).strip() else "." for r in refs]

    P, R, F1 = score(safe_preds, safe_refs, lang="en", model_type=model_type, verbose=verbose)
    
    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item()
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate BERTScore for model comparison.")
    parser.add_argument("--ref-file", required=True, help="官方测试集文件路径 (例如 rrgen_test_data.txt)")
    parser.add_argument("--core-file", required=True, help="CoRe 模型生成的预测文件")
    parser.add_argument("--model-type", default=llm_config.eval_model, help="用于计算的底层模型 (默认 roberta-large)")
    args = parser.parse_args()

    print("正在加载数据...")
    refs = load_texts(args.ref_file, is_ref=True)
    core_preds = load_texts(args.core_file, is_ref=False)

    # 确保行数绝对对齐，防止“行号错位灾难”
    n = min(len(refs), len(core_preds))
    refs = refs[:n]
    core_preds = core_preds[:n]
    
    assert len(refs) == len(core_preds), \
        f"数据行数不一致！Refs: {len(refs)}, CoRe: {len(core_preds)}"

    print(f"数据加载完毕，共对齐 {n} 条数据。")
    print(f"正在使用模型 {args.model_type} 计算 BERTScore...\n")

    # 计算 CoRe 模型的 BERTScore
    print("--------------------------------------------------")
    print("开始计算 CoRe 模型的 BERTScore (第一次运行会自动下载模型)...")
    bert_result = compute_bertscore(core_preds, refs, model_type=args.model_type, verbose=True)

    print("\n==================================================")
    print("BERTScore 最终评测结果")
    print("==================================================")
    print(f"{'模型':<10} | {'Precision (精确率)':<20} | {'Recall (召回率)':<20} | {'F1 (综合得分)':<20}")
    print("-" * 75)
    print(f"{'CoRe：':<10} | {bert_result['Precision']:<20.4f} | {bert_result['Recall']:<20.4f} | {bert_result['F1']:<20.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
    