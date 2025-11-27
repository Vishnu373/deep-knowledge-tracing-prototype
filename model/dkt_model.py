import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# creating 1D vector input for GRU
def create_input(skill_id, correct, num_skills = 8):
    # one hot encoding
    skill_id_enc = torch.nn.functional.one_hot(skill_id - 1, num_skills)
    correct_enc = torch.nn.functional.one_hot(correct, 2)

    # combined input
    gru_input = torch.cat([skill_id_enc, correct_enc], dim = -1).float()

    return gru_input

class DKTModel(nn.Module):
    def __init__(self, num_skills = 8, hidden_size = 128, sigma = 0.01):
        super(DKTModel, self).__init__()

        # Input layer
        self.input_size = 2 * num_skills
        
        # Output layer
        self.num_skills = num_skills
        
        # activation function
        self.sigmoid = nn.Sigmoid()

        # gru layer
        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = hidden_size,
            batch_first = True
        )

        # output layer
        self.fc = nn.Linear (
            hidden_size,
            num_skills
        )

    def forward(self, input):
        gru_out, _ = self.gru(input)
        logits = self.fc(gru_out)
        predictions = self.sigmoid(logits)

        return predictions
