# Meta Llama Reasoner

Welcome to the Meta Llama Reasoner repository! This project aims to provide the code necessary to train, evaluate and use a reasoner model based on the model `llama-3.3-70b-versatile`.

## Table of Contents

- [Brief explanation](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)

## Brief explanation

Meta Llama Reasoner is designed to make the base model `llama-3.3-70b-versatile` reason with a chain of thought before answering a question. A reward prediction model will be trained to be able to select the best chain of thought that the base model produces. 

This reward prediction model will be trained on a Mathematical dataset that contains answers and questions for complex problems. The rational steps that the base model produces and the question is being asked is the X. The Y is whether the final answer of the base model will be correct or not. This iterative process helps in refining the reasoning capabilities of the base model, making it more reliable and effective in generating correct answers.

## Installation

To install Meta Llama Reasoner, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/sergiperez73/meta-llama-reasoner.git
    ```
2. Navigate to the project directory:
    ```bash
    cd meta-llama-reasoner
    ```
3. Install the required dependencies:
    ```bash
    pip install requirements.txt
    ```
    Be sure to use Python version 3.10 to 3.12.

4. Donwload the following dataset from the Kaggle webpage: [AIME Problem Set: 1983-2024](https://www.kaggle.com/datasets/hemishveeraboina/aime-problem-set-1983-2024/data)

## Usage

The `llama-3.3-70b-versatile` model is used thanks to the Groq API. Therefore, you will need to specify your API key on the `config/.env` file before executing the code.

### Training process

To train the model for the first time, you need to execute a command like the following example.

```bash
python training.py --starting_question 0 --n_training_questions 10 --sleep_time 30 --lr 0.001 --n_calls_per_question 5 --layer_config_embedding "512 256" --layer_config_general "512 256 1" --model_name "my_model"
```

After the training process, a folder with the name specified on the `model_name` parameter will be created. It will contain:
* A dataset with the results on each question
* The metadata with the parameters used on the training execution
* A file containing the weights of the model after the training.

If you want to continue training your model, you can load it with the `--load_model_name` parameter, specifying there the name of the pretrained model you want to load. It is a good practice to fix the starting question to the next question of the last question the pretrained model answered, which can be extracted from the metadata of the pretrained model.

### Inference

Once you have trained a model, you can do inference with it. You will also need to prepare a bunch of questions you want to ask to the model. This questions need to be inside a folder on the `inf_questions_answers` folder. The name of the folder will identify the file contained on it that includes the required questions. This file will has to be named `questions_answers.json`. You can find an example on the file `inf_questions_answers/example_questions/questions_answers.json`.

Having created the file containing the questions, you can execute a command like the following example.

```bash
python inference.py --sleep_time 30 --load_model_name "1738612360" --folder_name "example_questions" --n_calls_per_question 10
```

The model will select the best possible answer among the answers the base model outputs. Therefore, the more calls per question you fix, the more likely is to find the best answer. The answers will appear on the same file the questions were gotten.

## Author

Sergi Pérez Escalante