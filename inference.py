import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset_loader.dataset_loader import getDataset
from models.reward_prediction import ModelRewardPrediction, get_embedding
from models.meta_llama_rational_steps import MetaLlamaRationalSteps
import os
import json

# Put questions in a json
# Sleep time between questions
# Output answers in a json

system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>"


def perform_inference(model_reward_accuracy, model_rational_steps, questions, args):
    results = []
    for question in questions:

        model_answer_list = []
        question_embedding_list = []
        rational_step_embedding_list = []
        for j in range(args.n_calls_per_question):
            rational_step, model_answer = model_rational_steps.question_steps_answer(question=question, system_prompt=system_prompt)
            
            question_embedding = torch.tensor(get_embedding(question)).unsqueeze(0)
            rational_step_embedding = torch.tensor(get_embedding(rational_step)).unsqueeze(0)

            model_answer_list.append("Rational Step: "+rational_step+"\nAnswer: "+model_answer)
            question_embedding_list.append(question_embedding)
            rational_step_embedding_list.append(rational_step_embedding)    
        
        question_embedding = torch.cat(question_embedding_list, dim=0)
        rational_step_embedding = torch.cat(rational_step_embedding_list, dim=0)

        outputs = model_reward_accuracy([question_embedding, rational_step_embedding])
        
        max_index = torch.argmax(outputs)
        results.append(model_answer_list[max_index])
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--n_questions', type=int, default=4, required=False, help='Number of questions')
    parser.add_argument('--load_model_name', type=str, default=None, required=True, help='Load a model to continue training')
    parser.add_argument('--n_calls_per_question', type=int, default=3, required=False, help='Number of calls per question')

    args = parser.parse_args()

    questions = [
        "Follow the sequence: 1 = 5, 2 = 10, 3 = 15, 4 = 20, 5 = ?",
        "Find a word that relates on the meaning 'Juicio' and 'Tristeza'"
    ]

    if args.load_model_name is None:
        raise ValueError("Please provide a model name to load")

    with open(f'trained_models/{args.load_model_name}/training_metadata.json', 'r') as f:
        training_metadata = json.load(f)

    model_reward_accuracy = ModelRewardPrediction(layer_config_embedding = training_metadata['layer_config_embedding'],
                                                    layer_config_general = training_metadata['layer_config_general'],
                                                    n_embeddings=2)
    model_reward_accuracy.load_state_dict(torch.load(f'trained_models/{args.load_model_name}/model_weights.pth', weights_only=True))

    model_reward_accuracy.eval()
    
    model_rational_steps = MetaLlamaRationalSteps()

    answers = perform_inference(model_reward_accuracy, model_rational_steps, questions, args)
    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"Question {i}:\n\n {question}\n\n{answer}\n\n\n")