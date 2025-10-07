import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from Models.model import MultiScoFeed
from utils import EssayDataset, load_leaf_dataset, score_2_prompt, score_2_prompt_overall
from evaluation import get_score_eval, get_feedback_eval
import argparse
import torch.nn.functional as F
from models.t5_sco_feed import T5ScoFeed

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    
def training(model, train_dl, mse_loss):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dl):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        scores = batch["scores"].to(device)
        features = batch["features"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, features=features)

        regression_loss = mse_loss(outputs["scores"], scores)
        generation_loss = outputs["loss"] # text generation loss

        total_batch_loss = generation_loss + regression_loss
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

def validation(model, tokenizer, val_dl):
    model.eval()
    pred_scores = []
    act_scores = []
    pred_feedback = []
    act_feedback = []
    for batch in tqdm(val_dl):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features = batch['features'].to(device)
        actual = batch["scores"]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, features=features)
            score = outputs["score_pred"].item()
            
            generated_ids = model.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        pred_scores.extend(score)
        act_scores.extend(actual)
        pred_feedback.extend(generated)
        act_feedback.extend(labels)
        
    return pd.DataFrame(
        {
            "predicted": pred_scores,
            "actual": act_scores,
            "candidates": pred_feedback,
            "references": act_feedback
        }
    )
        
def main():
    parser = argparse.ArgumentParser(description="Multi-trait score and Feedback generation T5 Model")
    parser.add_argument('--model', type=str, default="t5", help="Choose model type")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--seed', type=int, default=12, help="Select seed")
    parser.add_argument('--features', type=str, default="yes", help="Features included: yes or no")
    parser.add_argument('--lrate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--output', type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    return args.model, args.epochs, args.batch, args.seed, args.features, args.lrate, args.output

if __name__ == '__main__':
    model_args, epochs, batch_size, seed, features, learning_rate, output_dir = main()
    
    # Set seed
    set_seed(seed)
    
    # Set hyperparameter
    model_name="google/flan-t5-base" 
    num_traits = 5 
    feature_dim = 18
    model = T5ScoFeed(model_name, num_traits, feature_dim)
        
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load the dataset
    train, val, test = load_leaf_dataset()
    
    train_ds = EssayDataset(train['essay'].tolist(), train['scores'].tolist(), train['feedback'].tolist(), train['features'].tolist())
    val_ds = EssayDataset(val['essay'].tolist(), val['scores'].tolist(), val['feedback'].tolist(), val['features'].tolist())
    test_ds = EssayDataset(test['essay'].tolist(), test['scores'].tolist(), test['feedback'].tolist(), test['features'].tolist())
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    for epoch in range(1, epochs + 1):
        training(model, train_dl, optimizer, loss_fn, "Training")

        if epoch % 5 == 0:
            output_val = validation(model, val_dl, "Validation")

            score_val_eval = get_score_eval(output_val['predicted'], output_val['actual'])
            feedback_val_eval = get_feedback_eval(output_val['references'], output_val['candidates'])
            
            # Save model with more than 0.5 QWK
            if score_val_eval['qwk'] > 0.5:
                best_models_wts = model.state_dict().copy()
                torch.save(best_models_wts, f"{output_dir}/{model_args}-{batch_size}-{seed}-{features}-{learning_rate}-{epoch}.pth")
                
            tqdm.write(f"Epoch, {epoch}")
            tqdm.write(f"QWK: {score_val_eval['qwk']}, Acc: {score_val_eval['accuracy']}")
            tqdm.write(f"BertScore: {feedback_val_eval['bert']}, BLEU score: {feedback_val_eval['bleu']}, ROUGE-L: {feedback_val_eval['rougel']}, METEOR: {feedback_val_eval['meteor']}")

    # Collect test results
    output_test = validation(model, test_dl, "Testing")
    output = pd.DataFrame(
        {
            "actual": score_2_prompt(output_test['actual']), 
            "predicted": output_test['predicted'],
            "references": output_test['references'],
            "candidates": output_test['candidates']
        })
    output.to_csv(f"{output_dir}/test_pred-{model_args}-{batch_size}-{seed}-{features}-{learning_rate}.csv", index=False)

    score_test_eval = get_score_eval(output_test['predicted'], output_test['actual'])
    feedback_test_eval = get_feedback_eval(output_test['references'], output_test['candidates'])
    print(f'QWK test: {score_test_eval['qwk']}')
    print(f'Acc test: {score_test_eval['accuracy']}')
    print(f"BertScore: {feedback_test_eval['bert']}, BLEU score: {feedback_test_eval['bleu']}, ROUGE-L: {feedback_test_eval['rougel']}, METEOR: {feedback_test_eval['meteor']}")


    # Save model
    torch.save(model.state_dict(), f"{output_dir}/{model_args}-{batch_size}-{seed}-{features}-{learning_rate}.pth")
    
    
    