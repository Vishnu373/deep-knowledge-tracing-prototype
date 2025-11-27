from pydantic import BaseModel, Field
from typing import List

class Interaction(BaseModel):
    skill_id: int
    correct: int

class ModelInput(BaseModel):
    student_id: str
    history: List[Interaction]

class ModelOutput(BaseModel):
    next_skill_id: int
    predicted_success: float

