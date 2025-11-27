import numpy as np
import pandas as pd

# generating synthetic data
def generate_data(num_students = 100, max_seq_len = 20, num_skills = 8,):
    data = []

    for student_id in range(1, num_students + 1):
        num_ques_len = np.random.randint(5, max_seq_len)
        skill_seq = np.random.randint(1, num_skills + 1, num_ques_len)
        correctedness_seq = np.random.choice([0, 1], num_ques_len, p = [0.4, 0.6])

        for order in range(num_ques_len):
            data.append({
                "student_id": f"S{student_id:03d}",
                "time_stamp": order,
                "skill_id": int(skill_seq[order]),
                "correct": int(correctedness_seq[order])
            })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/synthetic_data.csv", index = False)
