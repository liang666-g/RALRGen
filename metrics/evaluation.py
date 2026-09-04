from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score


def compute_rouge(reference_corpus, translation_corpus):
    rouge = Rouge()
    rouge_1 = {'f': 0, 'p': 0, 'r': 0}
    rouge_2 = {'f': 0, 'p': 0, 'r': 0}
    rouge_l = {'f': 0, 'p': 0, 'r': 0}
    for i in range(len(reference_corpus)):
        scores = rouge.get_scores(translation_corpus[i], reference_corpus[i])
        rouge_1['f'] += scores[0]['rouge-1']['f']
        rouge_1['p'] += scores[0]['rouge-1']['p']
        rouge_1['r'] += scores[0]['rouge-1']['r']
        rouge_2['f'] += scores[0]['rouge-2']['f']
        rouge_2['p'] += scores[0]['rouge-2']['p']
        rouge_2['r'] += scores[0]['rouge-2']['r']
        rouge_l['f'] += scores[0]['rouge-l']['f']
        rouge_l['p'] += scores[0]['rouge-l']['p']
        rouge_l['r'] += scores[0]['rouge-l']['r']
    lens = len(reference_corpus)
    rouge_1['f'] = rouge_1['f'] / lens
    rouge_1['p'] = rouge_1['p'] / lens
    rouge_1['r'] = rouge_1['r'] / lens

    rouge_2['f'] = rouge_2['f'] / lens
    rouge_2['p'] = rouge_2['p'] / lens
    rouge_2['r'] = rouge_2['r'] / lens

    rouge_l['f'] = rouge_l['f'] / lens
    rouge_l['p'] = rouge_l['p'] / lens
    rouge_l['r'] = rouge_l['r'] / lens

    return {'rouge-1': {'f': rouge_1['f'], 'p': rouge_1['p'], 'r': rouge_1['r']},
            'rouge-2': {'f': rouge_2['f'], 'p': rouge_2['p'], 'r': rouge_2['r']},
            'rouge-l': {'f': rouge_l['f'], 'p': rouge_l['p'], 'r': rouge_l['r']}}


def compute_meteor(reference_corpus, translation_corpus):
    meteor = 0.0
    for i in range(len(reference_corpus)):
        ref = reference_corpus[i]
        hyp = translation_corpus[i]

        if isinstance(ref, str):
            ref = ref.split()
        if isinstance(hyp, str):
            hyp = hyp.split()

        cur_meteor = single_meteor_score(ref, hyp)
        meteor += cur_meteor

    meteor = meteor / len(reference_corpus)
    return meteor

