from fastapi import FastAPI
import torch
from model.dkt_model import DKTModel, create_input
from api.schemas import ModelInput, ModelOutput, Interaction
import traceback

app = FastAPI()

@app.get("/")
def root():
    return "Deep knowledge tracing prototype is running"

model = DKTModel()
model.load_state_dict(torch.load("model/dkt_model.pth"))
model.eval()

@app.post("/predict/", response_model=ModelOutput)
def predict_endpoint(model_input: ModelInput):
    try:
        # 1. unpacking the model input
        skill_ids = [interaction.skill_id for interaction in model_input.history]
        corrects = [interaction.correct for interaction in model_input.history]

        # 2. converting to torch tensors
        # Assuming input is 1-based (from user) and model needs 0-based
        skill_ids_tensor = torch.tensor(skill_ids) - 1
        corrects_tensor = torch.tensor(corrects)

        # 3. one single input for the model
        input = create_input(skill_ids_tensor, corrects_tensor)

        # 4. running the model
        with torch.no_grad():
            # Add batch dimension: [1, seq_len, input_size]
            input_tensor = input.unsqueeze(0)
            output = model(input_tensor)

        # 5. getting most up-to date predictions
        last_prediction = output[0, -1, :]  # Shape: [num_skills]
        
        # Get the last skill the student attempted
        last_skill_attempted = skill_ids[-1] - 1  # Convert to 0-indexed
        last_was_correct = corrects[-1]
        
        # Recommendation logic:
        # If they got it wrong, recommend practicing that skill again
        # If they got it right, recommend the skill with highest probability
        if last_was_correct == 0:
            # They failed - recommend the same skill
            next_skill_id = skill_ids[-1]  # Keep 1-indexed
            predicted_success = last_prediction[last_skill_attempted].item()
        else:
            # They succeeded - recommend next best skill
            next_skill_id = last_prediction.argmax().item() + 1
            predicted_success = last_prediction.max().item()

        return {
            "next_skill_id": next_skill_id,
            "predicted_success": predicted_success
        }
    except Exception as e:
        traceback.print_exc()
        raise e