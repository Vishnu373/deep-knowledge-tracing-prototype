import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from model.dkt_model import create_input, DKTModel
import torch.nn as nn

def data_preparation():
    # loading the file
    data = pd.read_csv("data/synthetic_data.csv")

    # grouping by student_id
    data_grouped = data.groupby(data["student_id"])

    # each index corresponds to all the details one student
    student_seq = []
    for student_id, group in data_grouped:
        student_seq.append(group)

    return student_seq

def data_split(sequences):
    train_seq, test_seq = train_test_split(
        sequences,
        test_size=0.2,
        random_state=1
    )

    return train_seq, test_seq

def seq_preparation(train_seq, num_skills = 8):
    X_list = []
    y_list = []

    for seq in train_seq:
        skill_ids = seq["skill_id"].values - 1
        corrects = seq["correct"].values

        # Skipping students with only 1 skill
        if len(skill_ids) < 2:
            continue
        
        # preparing the input
        X_skills = torch.tensor(skill_ids[:-1])
        X_corrects = torch.tensor(corrects[:-1])
        X = create_input(X_skills, X_corrects, num_skills)

        # preparing the target
        y_skills = torch.tensor(skill_ids[1:])
        y_corrects = torch.tensor(corrects[1:])

        # final input, target
        X_list.append(X)
        y_list.append((y_skills, y_corrects))

    return X_list, y_list

def train_model(X_train, y_train, num_skills=8, hidden_size=128, epochs=50, learning_rate=0.001):
    # 1. model
    model = DKTModel(num_skills, hidden_size)

    # 2. optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # 3. loss function - binary cross entropy
    loss_fn = nn.BCELoss()

    # 4. training loop
    for epoch in range(epochs):
        # training mode
        model.train()
        total_loss = 0

        for i in range(len(X_train)):
            X = X_train[i].unsqueeze(0)
            y_skills, y_corrects = y_train[i]

            # Forward pass
            predictions = model(X)
            pred_probs = predictions[0, range(len(y_skills)), y_skills]

            # applying loss function
            target = y_corrects.float()
            loss = loss_fn(pred_probs, target)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.3f}")

    # saving the model
    torch.save(model.state_dict(), "model/dkt_model.pth")
    
    return model


if __name__ == "__main__":
    sequences = data_preparation()
    train, test = data_split(sequences)
    X_train, y_train = seq_preparation(train)
    X_test, y_test = seq_preparation(test)
    
    train_model(X_train, y_train)