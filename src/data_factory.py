from __future__ import annotations
import re
import numpy as np
from tqdm import tqdm
from configuration import llm_config
from hf_model_utils import load_sentence_transformer
import torch


def sbert_build_similarity(
    srcfile,
    inputfile,
    outputfile,
    n,
    model_name=llm_config.embedding_model,
    batch_size=64,
    device=None,
    group_by_app: bool = True,
):
    """
    使用 Sentence-BERT 做语义相似度检索，输出索引文件（每行 n+1 个全局样本下标）。

    - group_by_app=True（默认）：按 app_id 分组，仅在同一 app 内检索。
    - group_by_app=False：在**整个语料库**中全局检索相似评论。
    每行写入 ``n+1`` 个全局样本下标（空格分隔），与 ``tester`` 读取的 simi 文件格式一致。
    """

    model = load_sentence_transformer(model_name, device=device)

    if group_by_app:
        # 旧行为：按 app 分组建池
        src_sent_app = {}
        response = {}
        i = 0
        with open(srcfile, encoding="utf-8", errors="replace") as f:
            for sent in tqdm(f.readlines(), desc="Load corpus (srcfile)"):
                terms = sent.split("-[split]-")
                if len(terms) < 8:
                    continue
                app_id = terms[0]
                src_sent = terms[4]
                if app_id not in src_sent_app:
                    src_sent_app[app_id] = []
                src_sent_app[app_id].append(src_sent)
                if app_id not in response:
                    response[app_id] = []
                response[app_id].append(i)
                i = i + 1

        for app, texts in src_sent_app.items():
            for j in range(len(texts)):
                texts[j] = re.sub(r"[.,!?<>():;\[\]]", " ", texts[j])

        emb_app = {}
        for app, texts in tqdm(list(src_sent_app.items()), desc="SBERT encode per app"):
            if not texts:
                continue
            emb = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            emb_app[app] = np.asarray(emb, dtype=np.float32)
    else:
        # 新行为：全语料库建一个大池（不按 app 分组）
        corpus_texts = []
        corpus_ids = []
        with open(srcfile, encoding="utf-8", errors="replace") as f:
            for sent in tqdm(f.readlines(), desc="Load corpus (srcfile)"):
                terms = sent.split("-[split]-")
                if len(terms) < 8:
                    continue
                src_sent = re.sub(r"[.,!?<>():;\[\]]", " ", terms[4])
                corpus_texts.append(src_sent)
                corpus_ids.append(len(corpus_ids))

        emb = model.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        corpus_mat = np.asarray(emb, dtype=np.float32)

    index = []
    m = 1
    with open(inputfile, encoding="utf-8", errors="replace") as f:
        for sent in f.readlines():
            terms = sent.split("-[split]-")
            if len(terms) < 8:
                continue
            src_sent = terms[4]
            src_sent = re.sub(r"[.,!?<>():;\[\]]", " ", src_sent)

            if group_by_app:
                app_id = terms[0]
                corpus_emb = emb_app.get(app_id)
                if corpus_emb is None or corpus_emb.size == 0:
                    index.append([])
                    continue
            else:
                corpus_emb = corpus_mat
            q = model.encode(
                [src_sent],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0].astype(np.float32)
            sims = corpus_emb @ q
            m_size = sims.shape[0]
            want = min(n + 1, m_size)
            if m_size <= n + 1:
                top_local = np.argsort(-sims)
            else:
                top_local = np.argpartition(-sims, want - 1)[:want]
                top_local = top_local[np.argsort(-sims[top_local])]

            if group_by_app:
                globs = [response[app_id][int(j)] for j in top_local]
            else:
                globs = [int(j) for j in top_local]
            # 按用户要求：不凑满 n 条，有多少写多少（0 条则写空行）
            index.append(globs[:want])

            if (m % 1000) == 0:
                print("完成" + str(m))
            m = m + 1

    with open(outputfile, "wt", encoding="utf-8", newline="") as f:
        for row in index:
            f.write(" ".join(str(t) for t in row))
            f.write("\n")
