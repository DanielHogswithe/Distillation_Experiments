import spacy
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader

# def clean(input_df: pd.DataFrame) -> pd.DataFrame:
#     input_df.replace(to_replace="<br /><br />", value="", regex=True, inplace=True)
#     input_df["label"] = input_df["sentiment"].apply(filter)
#     input_df.drop(columns=["sentiment"], inplace=True)

#     return input_df

# def filter(label: str):

#     if label == "positive":
#         return 1
#     else:
#         return 0

class MyIMDBData(Dataset):
    def __init__(self, csv_file, transform=None):   #   add root_dir argument if csv is in different folder
        self.annotations = pd.read_csv(csv_file)
        self.annotations.drop(columns=["Unnamed: 0"], inplace=True)
        # self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.annotations) #  Number of datapoints in the set

    def __getitem__(self, index):
        review = self.annotations.iloc[index, 1]
        sentiment = self.annotations.iloc[index, 2]
        return (review, sentiment)
        return 