import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path("student_data.csv")
OUTPUT_DIR = Path("outputs")


def generate_student_data(filename=DATA_FILE, seed=42):
    """Generate a realistic student dataset with attendance-grade correlation."""
    np.random.seed(seed)
    random.seed(seed)

    first_names = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
        "Sai", "Ayaan", "Krishna", "Ishaan", "Shaurya",
        "Ananya", "Aadhya", "Diya", "Saanvi", "Priya",
        "Neha", "Riya", "Kriti", "Pooja", "Kavya",
        "Rahul", "Rohan", "Amit", "Karan", "Vikram",
        "Sneha", "Nidhi", "Tanvi", "Rashi", "Meera"
    ]

    data = []

    for name in first_names:
        attendance = np.random.randint(60, 101)
        base_score = attendance * 0.8

        math_grade = min(
            100, max(0, int(np.random.normal(base_score, 10)))
        )
        science_grade = min(
            100, max(0, int(np.random.normal(base_score, 8)))
        )
        english_grade = min(
            100, max(0, int(np.random.normal(base_score + 5, 7)))
        )

        if random.random() < 0.05:
            math_grade = np.nan

        data.append([
            name,
            math_grade,
            science_grade,
            english_grade,
            attendance
        ])

    df = pd.DataFrame(
        data,
        columns=[
            "Student_Name",
            "Math_Score",
            "Science_Score",
            "English_Score",
            "Attendance_Pct"
        ]
    )

    df.to_csv(filename, index=False)
    print(f"Dataset generated and saved to: {filename}")
    return df


def load_and_clean_data(filename=DATA_FILE):
    """Load the CSV file and fill missing numeric values with column means."""
    if not Path(filename).exists():
        print("Dataset not found. Generating a new dataset...")
        df = generate_student_data(filename)
    else:
        df = pd.read_csv(filename)

    numeric_columns = [
        "Math_Score",
        "Science_Score",
        "English_Score",
        "Attendance_Pct"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].mean())

    df["Overall_Score"] = df[
        ["Math_Score", "Science_Score", "English_Score"]
    ].mean(axis=1)

    df["Result"] = np.where(df["Overall_Score"] >= 40, "Pass", "Fail")
    df["Grade"] = pd.cut(
        df["Overall_Score"],
        bins=[-np.inf, 40, 60, 80, 90, np.inf],
        labels=["F", "C", "B", "A", "A+"],
        right=False
    )

    return df


def perform_statistical_analysis(df):
    """Display mean, median, standard deviation and attendance correlation."""
    subjects = ["Math_Score", "Science_Score", "English_Score"]

    summary = df[subjects].agg(["mean", "median", "std"]).T
    summary.columns = ["Mean", "Median", "Standard_Deviation"]

    print("\nSTATISTICAL SUMMARY")
    print(summary.round(2))

    correlation = df["Attendance_Pct"].corr(df["Overall_Score"])
    print(f"\nAttendance vs Overall Score Correlation: {correlation:.2f}")

    return summary, correlation


def calculate_probability(df, threshold=80):
    """Calculate empirical probability of scoring above a threshold."""
    total_students = len(df)
    students_above_threshold = (df["Overall_Score"] > threshold).sum()
    probability = students_above_threshold / total_students

    pass_probability = (df["Result"] == "Pass").mean()

    print("\nPROBABILITY ANALYSIS")
    print(
        f"Probability of scoring above {threshold}%: "
        f"{probability:.2%}"
    )
    print(f"Probability of passing: {pass_probability:.2%}")

    return {
        "probability_above_threshold": probability,
        "pass_probability": pass_probability
    }


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two numeric vectors."""
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def create_study_buddy_pairs(df):
    """Match students with the most similar grade profiles."""
    subjects = ["Math_Score", "Science_Score", "English_Score"]
    students = df["Student_Name"].tolist()
    vectors = df[subjects].to_numpy(dtype=float)

    similarity_matrix = np.zeros((len(students), len(students)))

    for i in range(len(students)):
        for j in range(len(students)):
            similarity_matrix[i, j] = cosine_similarity(
                vectors[i], vectors[j]
            )

    available = set(range(len(students)))
    pairs = []

    while len(available) >= 2:
        student_i = min(available)
        available.remove(student_i)

        best_student_j = max(
            available,
            key=lambda j: similarity_matrix[student_i, j]
        )
        available.remove(best_student_j)

        pairs.append({
            "Student_1": students[student_i],
            "Student_2": students[best_student_j],
            "Cosine_Similarity": similarity_matrix[
                student_i, best_student_j
            ]
        })

    if available:
        pairs.append({
            "Student_1": students[min(available)],
            "Student_2": "No partner available",
            "Cosine_Similarity": np.nan
        })

    pairs_df = pd.DataFrame(pairs)

    print("\nSTUDY BUDDY PAIRINGS")
    print(pairs_df.to_string(index=False))

    return pairs_df


def create_visualizations(df, output_dir=OUTPUT_DIR):
    """Create subject-average and grade-distribution bar charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = ["Math_Score", "Science_Score", "English_Score"]
    subject_labels = ["Math", "Science", "English"]
    averages = df[subjects].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(subject_labels, averages)
    plt.title("Class Average by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Average Score")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "subject_averages.png", dpi=300)
    plt.close()

    grade_counts = df["Grade"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(grade_counts.index.astype(str), grade_counts.values)
    plt.title("Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "grade_distribution.png", dpi=300)
    plt.close()

    print(f"\nCharts saved in: {output_dir.resolve()}")


def save_reports(df, summary, probabilities, pairs_df, output_dir=OUTPUT_DIR):
    """Save analysis results as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "cleaned_student_data.csv", index=False)
    summary.to_csv(output_dir / "statistical_summary.csv")
    pd.DataFrame([probabilities]).to_csv(
        output_dir / "probability_summary.csv",
        index=False
    )
    pairs_df.to_csv(output_dir / "study_buddy_pairs.csv", index=False)

    print(f"Reports saved in: {output_dir.resolve()}")


def main():
    print("=" * 60)
    print("STUDENT PERFORMANCE & STUDY GROUP ANALYZER")
    print("=" * 60)

    df = load_and_clean_data()

    print("\nFIRST FIVE STUDENT RECORDS")
    print(df.head())

    summary, correlation = perform_statistical_analysis(df)
    probabilities = calculate_probability(df, threshold=80)
    pairs_df = create_study_buddy_pairs(df)
    create_visualizations(df)
    save_reports(df, summary, probabilities, pairs_df)

    print("\nProject execution completed successfully.")


if __name__ == "__main__":
    main()
