import pandas as pd


def getDataset():
    AIME_Dataset = pd.read_csv("data/AIME_Dataset_1983_2024.csv")
    AIME_Dataset = AIME_Dataset[['Question', 'Answer']]
    AIME_Dataset_clarification_prompt = " The Answer must only contain a nummber, not an explanation."

    return AIME_Dataset, AIME_Dataset_clarification_prompt