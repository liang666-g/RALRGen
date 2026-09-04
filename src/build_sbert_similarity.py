from __future__ import annotations
import argparse
from data_factory import sbert_build_similarity
from configuration import llm_config


def main():
    p = argparse.ArgumentParser(description="Sentence-BERT RAG 相似评论索引")
    p.add_argument("--src", required=True, help="构建各 app 语料池用的数据文件（通常与训练集相同）")
    p.add_argument("--input", required=True, help="要对每一行生成邻居索引的文件（train/valid/test）")
    p.add_argument("--output", required=True, help="输出索引文件路径")
    p.add_argument(
        "-n",
        type=int,
        default=5,
        help="每行写入 n+1 个全局下标（建议 n >= configuration.runfig.rag_top_k）",
    )
    p.add_argument("--model", default=llm_config.embedding_model)
    p.add_argument("--batch-size", type=int, default=64, help="编码 batch 大小")
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    p.add_argument("--global-search", action="store_true", help="不按 app_id 分组，全语料库检索相似评论")
    args = p.parse_args()

    sbert_build_similarity(
        srcfile=args.src,
        inputfile=args.input,
        outputfile=args.output,
        n=args.n,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        group_by_app=not bool(args.global_search),
    )


if __name__ == "__main__":
    main()
