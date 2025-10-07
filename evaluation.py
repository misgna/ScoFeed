import evaluate
from sklearn.metrics import cohen_kappa_score
from utils import score_2_prompt
import numpy as np

bertscore = evaluate.load("bertscore")
rougescore = evaluate.load("rouge")
bleuscore = evaluate.load("bleu")
meteorscore = evaluate.load("meteor")

def get_score_eval(outputs, scores):
    predicted = score_2_prompt(outputs)
    actual = score_2_prompt(scores)
    return {
        "qwk": np.round(cohen_kappa_score(predicted, actual, weights='quadratic'), 3),
        "accuracy": np.round((np.array(predicted) == np.array(actual)).mean(), 3)
    }

def get_feedback_eval(references, generated):
    references_frmted = [[reference] for reference in references]
    bert = bertscore.compute(predictions=generated, references=references, model_type="roberta-large-mnli")
    rouge = rougescore.compute(predictions=generated, references=references)
    bleu = bleuscore.compute(predictions=generated, references=references_frmted)
    meteor = meteorscore.compute(predictions=generated, references=references_frmted)
    
    return {
        "bert": np.round(np.mean(bert['f1']), 3), 
        "rouge1": np.round(rouge['rouge1'], 3), 
        "rougel": np.round(rouge['rougeL'], 3),
        "bleu": np.round(bleu['bleu'], 3), 
        "meteor": np.round(meteor['meteor'], 3)
    }