# Deep Knowledge Tracing Prototype

A machine learning system that predicts student performance on educational tasks using Deep Knowledge Tracing (DKT). The model tracks a student's knowledge state over time based on their interaction history and predicts their likelihood of success on future questions.

## How It Works

```
Student History → Data Preprocessing → GRU Model → Prediction → Recommendation
(skill_id, correct)   (one-hot encoding)   (hidden state)   (probability)   (next skill)
```

**Architecture Flow:**
1. **Input**: Student interaction history (which skills attempted, correct/incorrect)
2. **Encoding**: Convert interactions to one-hot encoded vectors
3. **GRU Processing**: Recurrent neural network processes sequence and maintains knowledge state
4. **Output**: Probability of success for each skill
5. **Recommendation**: Suggest next skill based on performance (remediation or advancement)

## Project Structure

```
deep-knowledge-tracing-prototype/
├── model/
│   ├── dkt_model.py          # GRU model architecture
│   ├── train.py              # Training script
│   ├── evaluate.py           # Model evaluation
│   └── dkt_model.pth         # Trained model weights
├── api/
│   ├── main.py               # FastAPI server
│   └── schemas.py            # Pydantic data models
├── frontend/
│   └── app.py                # Streamlit web interface
├── data/
│   └── synthetic_data.csv    # Training data
└── requirements.txt
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Clone/Download the Project
```bash
cd deep-knowledge-tracing-prototype
```

### 3. Create Virtual Environment (Recommended)
```bash
python -m venv env
```

**Activate the environment:**
- Windows: `env\Scripts\activate`
- Mac/Linux: `source env/bin/activate`

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `torch` - PyTorch for the GRU model
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `streamlit` - Web UI framework
- `pandas` - data reading (in this use case)
- `scikit-learn` - Train/test splitting
- `requests` - HTTP client

### 5. Train the Model (Optional - model already trained)
```bash
python model/train.py
```

This will:
- Load synthetic student data
- Train the GRU model for 50 epochs
- Save weights to `model/dkt_model.pth`

### 6. Start the API Server
Open a terminal and run:
```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 7. Launch the Frontend
Open a **new terminal** (keep the API running) and run:
```bash
streamlit run frontend/app.py
```

The web interface will open automatically at `http://localhost:8501`

### 8. Test the API
Two inputs are required:
   - skill_id: range from 1 to 8 (1 to 8 some subjects or technical or topics or soft skills)
   - correct: 1 for correct, 0 for incorrect
- Basically the past performance of student.
## Usage

1. **Add Interactions**: Use the sidebar to add student interactions (Skill ID 1-8, Correct/Incorrect)
2. **View History**: See all interactions in the table on the left
3. **Get Prediction**: Click "Predict Next Performance" to see:
   - Recommended next skill
   - Probability of success
   - Color-coded recommendation (practice/advance/review)

**API not responding:**
- Ensure `uvicorn api.main:app --reload` is running
- Check that port 8000 is not in use

**Frontend can't connect:**
- Verify API is running at `http://127.0.0.1:8000`
- Check browser console for errors

**Model predictions seem off:**
- Model is trained on synthetic data with optimistic predictions
- Try adding more interactions to see probability changes
- Repeated failures on the same skill should lower predictions

**About the synthetic data:**
- Synthetic data was created from a portion of bigger dataset.
- Orginal dataset source: Assistments 2009-2010 (https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data?authuser=0)
