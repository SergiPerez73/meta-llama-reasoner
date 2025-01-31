import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset_loader.dataset_loader import getDataset
from models.reward_prediction import ModelRewardPrediction, get_embedding
from models.meta_llama_rational_steps import MetaLlamaRationalSteps

system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>"

# Put a starting question
# Be able to load the model saved

def training_loop(model_reward_accuracy, model_rational_steps, AIME_Dataset, clarification_prompt, args):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_reward_accuracy.parameters(), lr=args.lr)

    training_dict = {
        'Question_number': [],
        'Question': [],
        'Score': [],
        'Loss': []
    }

    for i in range(args.n_training_questions):
        print(f'Question {i+1}/{args.n_training_questions}')
        question = AIME_Dataset.iloc[i]['Question']
        answer = AIME_Dataset.iloc[i]['Answer']

        question_embedding_list = []
        rational_step_embedding_list = []
        score_list = []
        for j in range(args.n_calls_per_question):
            rational_step, model_answer = model_rational_steps.question_steps_answer(question=question, system_prompt=system_prompt+clarification_prompt)
            model_answer = model_answer.replace(" ", "")

            score = 0
            if model_answer == answer:
                score = 1
            
            score = torch.tensor(score).float().view(1, 1)
            question_embedding = torch.tensor(get_embedding(question)).unsqueeze(0)
            rational_step_embedding = torch.tensor(get_embedding(rational_step)).unsqueeze(0)

            score_list.append(score)
            question_embedding_list.append(question_embedding)
            rational_step_embedding_list.append(rational_step_embedding)    
        
        score = torch.cat(score_list, dim=0)
        question_embedding = torch.cat(question_embedding_list, dim=0)
        rational_step_embedding = torch.cat(rational_step_embedding_list, dim=0)

        optimizer.zero_grad()
        outputs = model_reward_accuracy([question_embedding, rational_step_embedding])
        loss = criterion(outputs, score)
        loss.backward()
        optimizer.step()

        training_dict['Question_number'].append(i)
        training_dict['Question'].append(question)
        training_dict['Score'].append(score.mean().item())
        training_dict['Loss'].append(loss.item())

        time.sleep(args.sleep_time)
    
    training_df = pd.DataFrame(training_dict)

    return model_reward_accuracy, training_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--n_training_questions', type=int, default=4, required=False, help='Number of training questions')
    parser.add_argument('--sleep_time', type=int, default=60, required=False, help='Seconds sleeping between questions (to avoid rate limiting)')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='Learning rate')
    parser.add_argument('--n_calls_per_question', type=int, default=3, required=False, help='Number of calls per question')
    parser.add_argument('--layer_config_embedding', type=str, default="512 256", required=False, help='Layer configuration for embeddings')
    parser.add_argument('--layer_config_general', type=str, default="256 128 1", required=False, help='Layer configuration for general layers')
    parser.add_argument('--model_name', type=str, default=f'{int(time.time())}', required=False, help='Name of the trained model')

    args = parser.parse_args()

    AIME_Dataset, clarification_prompt = getDataset()

    print('Dataset loaded')

    layer_config_embedding = args.layer_config_embedding
    layer_config_general = args.layer_config_general
    model_reward_accuracy = ModelRewardPrediction(layer_config_embedding, layer_config_general, n_embeddings=2)

    model_rational_steps = MetaLlamaRationalSteps()

    print('Models loaded')

    model_reward_accuracy, training_df = training_loop(model_reward_accuracy, model_rational_steps, AIME_Dataset, clarification_prompt, args)

    print('Training complete')

    torch.save(model_reward_accuracy.state_dict(), f'trained_models/trained_model_{args.model_name}.pth')
    training_df.to_parquet(f'training_log/training_log_{args.model_name}.parquet')

    print('Model and results saved')
