from model.dkt_model import DKTModel
import torch
from model.train import data_preparation, data_split, seq_preparation
from sklearn.metrics import roc_auc_score

def run_pipeline():
    # 1. create an instance of the model
    model = DKTModel(num_skills = 8, hidden_size = 128)

    # 2. load the model
    model.load_state_dict(torch.load("model/dkt_model.pth"))

    # 3. setting it to evaluation mode
    model.eval()

    # 4. preparing the test data
    sequences = data_preparation()
    _, test = data_split(sequences)
    X_test, y_test = seq_preparation(test)

    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for i in range(len(X_test)):
            X = X_test[i].unsqueeze(0)
            y_skills, y_corrects = y_test[i]

            # 5. get predictions
            predictions = model(X)
            pred_probs = predictions[0, range(len(y_skills)), y_skills]

            all_predictions.extend(pred_probs.tolist())
            all_actuals.extend(y_corrects.tolist())

    pred_binary = [1 if p > 0.5 else 0 for p in all_predictions]
    correct = sum([1 for pred, actual in zip(pred_binary, all_actuals) if pred == actual])
    
    # 6. get accuracy, auc-roc curve
    accuracy = correct / len(all_actuals)   
    auc = roc_auc_score(all_actuals, all_predictions)             

    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC-ROC: {auc:.3f}")

run_pipeline()