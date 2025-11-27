import streamlit as st
import requests
import pandas as pd

st.title("Deep Knowledge Tracing Prototype")
st.markdown("Predict student performance based on their interaction history.")

API_URL = "http://127.0.0.1:8000/predict/"
NUM_SKILLS = 8

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Add Interaction")

    # options for sidebar inputs
    skill_id = st.number_input("Skill ID", min_value=1, max_value=NUM_SKILLS, value=1)
    correct = st.selectbox("Outcome", options=[1, 0], format_func=lambda x: "Correct (1)" if x == 1 else "Incorrect (0)")
    
    # adds content to history when clicked
    if st.button("Add to History"):
        st.session_state.history.append({"skill_id": skill_id, "correct": correct})
        st.success(f"Added: Skill {skill_id}, {'Correct' if correct else 'Incorrect'}")

col1, col2 = st.columns([2, 1])

# left column
with col1:
    # table representation of history
    st.subheader("Student History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No history yet. Add interactions from the sidebar.")

# right column
with col2:
    st.subheader("Prediction")
    
    if st.button("Predict Next Performance", type="primary", disabled=len(st.session_state.history) == 0):
        try:
            # prepare payload for API request
            payload = {
                "student_id": "SessionUser",
                "history": st.session_state.history
            }
            
            # call predict api
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                pred_success = result["predicted_success"]
                next_skill = result["next_skill_id"]
                
                st.metric(label=f"Predicted Success for Skill {next_skill}", value=f"{pred_success:.1%}")
                st.progress(pred_success)
                
                if pred_success > 0.7:
                    st.success("High chance of success! Ready for harder content.")

                elif pred_success > 0.4:
                    st.warning("Moderate chance. Good for practice.")
                else:
                    st.error("Low chance. Review recommended.")
            
            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Connection Error: {e}")
