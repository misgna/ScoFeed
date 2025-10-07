from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from features import get_features
import joblib

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

class EssayDataset(Dataset):
    def __init__(self, essays, scores, feedback, features):
        self.essay_encodings = tokenizer(essays, padding=True, truncation=True, return_tensors='pt')
        self.feedback_encodings = tokenizer(feedback, padding=True, truncation=True, return_tensors='pt')
        self.scores = torch.tensor(scores, dtype=torch.float)
        self.features = torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        essay_encodings = {key: val[idx] for key, val in self.essay_encodings.items()}
        feedback_encodings = {key: val[idx] for key, val in self.feedback_encodings.items()}

        return {
            "input_ids": essay_encodings["input_ids"],
            "attention_mask": essay_encodings["attention_mask"],
            "labels": feedback_encodings["input_ids"],
            "scores": self.scores[idx],
            "features": self.features[idx]
        }
    
def load_leaf_dataset():
    filepath = "dataset/leaf++"
    
    train = pd.read_csv(os.path.join(filepath, 'train.csv'))
    #train['scores'] = train.apply(lambda row: [row['alignment_with_topic'], row['spelling_grammar_style'], row['clarity_of_view_point'], row['arguments_supporting_details']], axis=1)
    train['score_all'] = train[['alignment_with_topic', 'spelling_grammar_style', 'clarity_of_view_point', 'arguments_supporting_details']].sum(axis=1)
    train['score'] = train['score_all'].apply(score_2_minmax_overall)
    train['scores'] = train.apply(lambda row: [row['alignment_with_topic'].apply(score_2_minmax), row['spelling_grammar_style'].apply(score_2_minmax), row['clarity_of_view_point'].apply(score_2_minmax), row['arguments_supporting_details'].apply(score_2_minmax), row['score']], axis=1)
    

    dev = pd.read_csv(os.path.join(filepath, 'dev.csv'))
    #dev['scores'] = dev.apply(lambda row: [row['alignment_with_topic'], row['spelling_grammar_style'], row['clarity_of_view_point'], row['arguments_supporting_details']], axis=1)
    dev['score_all'] = dev[['alignment_with_topic', 'spelling_grammar_style', 'clarity_of_view_point', 'arguments_supporting_details']].sum(axis=1)
    dev['score'] = dev['score_all'].apply(score_2_minmax_overall)
    dev['scores'] = dev.apply(lambda row: [row['alignment_with_topic'].apply(score_2_minmax), row['spelling_grammar_style'].apply(score_2_minmax), row['clarity_of_view_point'].apply(score_2_minmax), row['arguments_supporting_details'].apply(score_2_minmax), row['score']], axis=1)
    

    test = pd.read_csv(os.path.join(filepath, 'test.csv'))
    #test['scores'] = test.apply(lambda row: [row['alignment_with_topic'], row['spelling_grammar_style'], row['clarity_of_view_point'], row['arguments_supporting_details'], row['score']], axis=1)
    test['score_all'] = test[['alignment_with_topic', 'spelling_grammar_style', 'clarity_of_view_point', 'arguments_supporting_details']].sum(axis=1)
    test['score'] = test['score_all'].apply(score_2_minmax_overall)
    test['scores'] = test.apply(lambda row: [row['alignment_with_topic'].apply(score_2_minmax), row['spelling_grammar_style'].apply(score_2_minmax), row['clarity_of_view_point'].apply(score_2_minmax), row['arguments_supporting_details'].apply(score_2_minmax), row['score']], axis=1)

    # Add manual features
    train = manual_fe(train, split='train')
    dev = manual_fe(dev)
    test = manual_fe(test)

    return train, dev, test

def manual_fe(dataset_df, split=''):
    features_df = dataset_df['essay'].apply(get_features).apply(pd.Series)
    normalized_features = []
    if split == 'train':
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features_df)
        joblib.dump(scaler, "output/feature_scaler.pkl")
    else:    
        scaler = joblib.load("output/feature_scaler.pkl")
        normalized_features = scaler.transform(features_df)
    
    dataset_df['features'] = normalized_features.tolist()
    return dataset_df

def score_2_minmax(scores):
    minimum = 1
    maximum = 5
    return (np.array(scores) - minimum) / (maximum - minimum)

def score_2_prompt(scores):
    minimum = 1
    maximum = 5
    return np.round(np.array(scores) * (maximum - minimum) + minimum).astype(int)

def score_2_minmax_overall(scores):
    minimum = 4
    maximum = 20
    return (np.array(scores) - minimum) / (maximum - minimum)

def score_2_prompt_overall(scores):
    minimum = 4
    maximum = 20
    return np.round(np.array(scores) * (maximum - minimum) + minimum).astype(int)