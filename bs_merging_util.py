from bs_merge_and_test import load_results, merge_result


def compute_merged_result(method):
    res1 = load_results('reference')
    num = len(res1)
    res2 = load_results(method)
    # res2 = load_results('vae_ia_rl_mlb_r2')
    res_file, mean_vqa_score, mean_unk_count = merge_result(res1, res2)
    from eval_vqa_question_oracle import evaluate_oracle
    scores = evaluate_oracle(res_file)

    return scores[1], mean_vqa_score
