import argparse
import os
import time
from datetime import datetime
from multiprocessing import freeze_support
import json
from configuration import file_space, llm_config
from tester import _load_test_data, _valid_test_llm
from metrics.bleu import compute_bleu
from metrics.evaluation import compute_rouge, compute_meteor
from metrics.BERTScore import compute_bertscore
from eval_data_cleaner import (
    clean_pred_and_test_for_eval,
    load_clean_predictions,
    write_metrics_and_cleanup,
)


def _safe_model_tag(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _percent_metric_values(obj, ndigits=2):
    if isinstance(obj, float):
        return round(obj * 100, ndigits)

    if isinstance(obj, list):
        return [_percent_metric_values(x, ndigits) for x in obj]

    if isinstance(obj, tuple):
        return [_percent_metric_values(x, ndigits) for x in obj]

    if isinstance(obj, dict):
        return {
            k: _percent_metric_values(v, ndigits)
            for k, v in obj.items()
        }

    return obj


def _parse_api_key_pool(text: str):
    """
    将命令行传入的多个 API Key 字符串解析成 list。
    支持英文逗号分隔，例如：
    sk-xxx,sk-yyy,sk-zzz
    """
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    start_time = time.time()
    print("CoRe — LLM 推理模式（无训练）。模型:", llm_config.model)

    p = argparse.ArgumentParser(description="CoRe LLM inference & evaluation (RAG + Qwen)")
    p.add_argument("--train-file", default=file_space.train_file, help="RAG 语料库（用于索引还原）")
    p.add_argument("--test-file", default=file_space.test_file, help="测试集文件")
    p.add_argument("--desc-features", default=file_space.desc_features, help="App 简介文件")
    p.add_argument("--test-simi-file", default=file_space.test_simi_file, help="测试集相似索引文件（可变长行）。若使用 FAISS 在线检索，可传空字符串。")
    p.add_argument("--no-test-simi", action="store_true", help="不使用预生成相似索引，启用FAISS 模式。")
    p.add_argument("--outtext-fp", default=llm_config.outtext_fp, help="输出目录根路径")
    p.add_argument("--worker-id", type=str, default="", help="手动指定进程后缀，区分并行任务")
    p.add_argument("--model", type=str, default=llm_config.model, help="本次运行使用的大模型名称")
    p.add_argument("--provider", type=str, default=llm_config.provider, choices=["openai", "dashscope"], help="本次运行使用的模型服务商")
    p.add_argument("--openai-base-url", type=str, default=llm_config.openai_base_url, help="OpenAI 兼容接口的 base_url")
    p.add_argument("--api-key-pool", type=str, default="", help="本次运行使用的 API Key 池，多个 key 用英文逗号分隔")
    p.add_argument("--max-workers", type=int, default=llm_config.max_workers, help="本次运行使用的最大进程数量")
    p.add_argument(
        "--task", 
        type=str, 
        choices=["generate", "evaluate", "both"], 
        default="both", 
        help="执行的任务：generate(仅生成回复), evaluate(仅算指标), both(生成后算指标)"
    )
    p.add_argument(
        "--pred-file", 
        type=str, 
        default="", 
        help="仅在 '--task evaluate' 时使用，指定要计算指标的模型预测文件路径 (如 stream_predictions.txt)"
    )
    args = p.parse_args()

    llm_config.model = args.model.strip() if args.model and args.model.strip() else llm_config.model
    llm_config.provider = args.provider.strip().lower() if args.provider and args.provider.strip() else llm_config.provider
    llm_config.openai_base_url = (
        args.openai_base_url.strip()
        if args.openai_base_url and args.openai_base_url.strip()
        else llm_config.openai_base_url
    )

    runtime_key_pool = _parse_api_key_pool(args.api_key_pool)
    llm_config.api_key_pool = runtime_key_pool if runtime_key_pool else llm_config.api_key_pool
    llm_config.max_workers = args.max_workers

    print(
        f"— 运行模式: {args.task} | "
        f"provider: {llm_config.provider} | "
        f"模型: {llm_config.model} | "
        f"base_url: {getattr(llm_config, 'openai_base_url', '')} | "
        f"api_key来源: {'api_key_pool' if getattr(llm_config, 'api_key_pool', []) else getattr(llm_config, 'api_key_env', '')}"
    )

    test_simi_path = "" if args.no_test_simi else args.test_simi_file

    test_app_ids, test_scores, test_src_texts, test_tgt_texts, test_desc_texts, test_simi_texts, test_extra_review_texts, test_review_times, test_reply_times, test_emotions = _load_test_data(
        args.test_file,
        args.desc_features,
        test_simi_path,
        args.train_file,
    )

    out_texts = []
    folder = ""
    eval_pred_file = ""
    unique_tag = args.worker_id

    # ================= 任务1：模型推理生成 =================
    if args.task in ["generate", "both"]:
        folder = os.path.join(
            args.outtext_fp,
            "llm_{}_com_{}_log_{}_{}".format(
                _safe_model_tag(llm_config.model),
                llm_config.comment_faiss_top_k,
                llm_config.log_rag_top_k,
                datetime.now().strftime("%Y%m%d_%H_%M_%S"),
            ),
        )
        os.makedirs(folder, exist_ok=True)

        t0 = time.time()
        out_texts = _valid_test_llm(
            test_app_ids,
            test_scores,
            test_src_texts,
            test_tgt_texts,
            test_desc_texts,
            test_simi_texts,
            test_extra_review_texts,
            test_review_times,
            test_reply_times,
            test_emotions,
            stream_save_dir=folder,
            file_suffix=unique_tag
        )

        eval_pred_file = os.path.join(folder, "predictions.txt")
        with open(eval_pred_file, "w", encoding="utf-8") as f:
            for i, reply in enumerate(out_texts):
                f.write(test_src_texts[i] + "-[split]-" + reply + "\n")

        print(f"✅ 已生成最终预测文件: {eval_pred_file}")
        print("推理与生成回复耗时(小时):", (time.time() - t0) / 3600.0)


    # ================= 任务2：指标评测计算 =================
    if args.task in ["evaluate", "both"]:
        # 如果是单读 evaluate 任务，需要从文件中解析 out_texts
        if args.task == "evaluate":
            if not args.pred_file or not os.path.exists(args.pred_file):
                raise ValueError("执行纯 evaluate 任务时，必须通过 --pred-file 指定已生成的预测文件路径！")
            eval_pred_file = args.pred_file

            folder = os.path.dirname(os.path.abspath(eval_pred_file))

        clean_result = clean_pred_and_test_for_eval(
            pred_file=eval_pred_file,
            test_file=args.test_file,
            out_dir=folder,
            sep="-[split]-",
        )

        test_app_ids, test_scores, test_src_texts, test_tgt_texts, test_desc_texts, test_simi_texts, test_extra_review_texts, test_review_times, test_reply_times, _test_emotions = _load_test_data(
            clean_result["final_test_file"],
            args.desc_features,
            test_simi_path,
            args.train_file,
        )

        out_texts = load_clean_predictions(clean_result["final_pred_file"], sep="-[split]-")

        # 提取原回复
        n = len(out_texts)
        tgt_slice = test_tgt_texts[:n]

        print(f"============== 开始指标评测计算 (共 {n} 条) ==============")
        # safe_out_texts = [out if out.strip() else "empty" for out in out_texts]

        references = [[t.strip().split()] for t in tgt_slice]
        candidates = [o.split() for o in out_texts]

        # 计算指标
        print("正在计算 BLEU, ROUGE, METEOR...")
        bleu_score, pls, _, _, _, _ = compute_bleu(references, candidates)
        # rouge_score = compute_rouge(tgt_slice, safe_out_texts)
        # meteor_score = compute_meteor(tgt_slice, safe_out_texts)
        rouge_score = compute_rouge(tgt_slice, out_texts)
        meteor_score = compute_meteor(tgt_slice, out_texts)

        # eval_model = getattr(llm_config, "eval_model", "roberta-large")
        # print(f"正在计算 BERTScore (模型: {eval_model})...")
        # bert_scores = compute_bertscore(out_texts, tgt_slice, model_type=eval_model, verbose=False)

        metrics_file = os.path.join(folder, "metrics.json")
        metrics_data = _percent_metric_values({
            "BLEU-4": bleu_score,
            "PLS": pls,
            "ROUGE": rouge_score, 
            "METEOR": meteor_score,
            # "BERTScore": bert_scores
        }, ndigits=2)

        write_metrics_and_cleanup(
            metrics_file=metrics_file,
            metrics_data=metrics_data,
            clean_result=clean_result,
            remove_intermediate_files=True,
        )

        # with open(metrics_file, "w", encoding="utf-8") as f:
        #     json.dump(metrics_data, f, ensure_ascii=False, indent=4)

        # with open(os.path.join(folder, "bleu"), "w", encoding="utf-8") as f:
        #     f.write(str(bleu_score))
        # with open(os.path.join(folder, "rouge"), "w", encoding="utf-8") as f:
        #     f.write(str(rouge_score))
        # with open(os.path.join(folder, "meteor"), "w", encoding="utf-8") as f:
        #     f.write(str(meteor_score))
        # with open(os.path.join(folder, "pls"), "w", encoding="utf-8") as f:
        #     for p in pls:
        #         f.write(str(p) + "\n")

        print("BLEU-4:", metrics_data["BLEU-4"])
        print("ROUGE:", metrics_data["ROUGE"])
        print("METEOR:", metrics_data["METEOR"])
        print("PLS:", metrics_data["PLS"])
        # print("BERTScore:", metrics_data["BERTScore"])
        print(f"评测结果已保存至: {metrics_file}")


if __name__ == "__main__":
    freeze_support()
    main()
