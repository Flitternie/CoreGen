from bleu.bleu import Bleu
# from cider.cider import Cider
from meteor.meteor import Meteor
import sys
from rouge import FilesRouge


def evaluate(hyp, ref):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [" ".join(v.strip().lower().split())] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        gts = {k: [v.strip().lower()] for k, v in enumerate(references)}
    # score_Bleu , stderr = Bleu().compute_score(hyp, ref)
    # print("Bleu_4: " + str(score_Bleu))

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor

    files_rouge = FilesRouge(hyp, ref)
    scores = files_rouge.get_scores(avg=True)
    print('Rouge: ' + str(scores))
    
    # score_Cider, scores_Cider = Cider().compute_score(gts, res)
    # print("Cider: "), score_Cider


if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])
