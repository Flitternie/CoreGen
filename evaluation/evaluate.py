from meteor.meteor import Meteor
from rouge import Rouge
import sys

def evaluate(ref, hyp):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [" ".join(v.strip().lower().split())] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        gts = {k: [v.strip().lower()] for k, v in enumerate(references)}
    
    meteor = Meteor()
    score_Meteor = meteor.compute_score(gts, res)
    print("Meteor: " + str(score_Meteor))

    rouge = Rouge()
    scores_Rouge = rouge.get_scores([i[0] for i in gts.values()], [i[0] for i in res.values()], avg=True)
    print("Rouge: " + str(scores_Rouge))
    

if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])
