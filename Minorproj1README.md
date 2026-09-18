# Student Performance & Study Group Analyzer

A beginner-friendly Python data analysis project that studies student marks and attendance and creates study-buddy pairings using Cosine Similarity.

## Features

- Generates a simulated dataset of 30 students.
- Includes Math, Science, English, and Attendance data.
- Handles missing values.
- Calculates:
  - Mean
  - Median
  - Standard deviation
  - Attendance-performance correlation
  - Probability of passing
  - Probability of scoring above 80%
- Creates study-buddy pairs using Cosine Similarity.
- Generates bar charts.
- Saves cleaned data and analysis reports.

## Project Structure

```text
student_performance_study_group_analyzer/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── student_data.csv              # generated after running
└── outputs/                      # generated after running
    ├── cleaned_student_data.csv
    ├── statistical_summary.csv
    ├── probability_summary.csv
    ├── study_buddy_pairs.csv
    ├── subject_averages.png
    └── grade_distribution.png
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

If `student_data.csv` does not exist, the program automatically generates it.

## Mathematical Concepts Used

### Mean

The average score of students in a subject.

### Standard Deviation

Measures the spread of student marks.

### Probability

The empirical probability is calculated as:

```text
Probability = Number of students satisfying condition / Total students
```

### Cosine Similarity

Cosine Similarity compares two student grade vectors:

```text
Cosine Similarity = A · B / (||A|| × ||B||)
```

A higher value means the students have more similar grade profiles.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Statistics
- Probability
- Linear Algebra
