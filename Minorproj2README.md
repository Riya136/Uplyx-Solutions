# Advanced Applications in Math and Python

This project implements the topics from the internship assignment:

- Covariance and correlation matrices
- Manual covariance calculation using matrix multiplication
- Verification using `numpy.cov`
- Markov chain simulation
- Steady-state probabilities using eigenvectors
- Time-series rolling mean and expanding standard deviation
- Advanced Matplotlib visualization using subplots and shading

## Project Structure

```text
advanced_math_python/
│
├── main.py
├── requirements.txt
├── README.md
├── advanced_math_python.ipynb
├── .gitignore
└── outputs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

The program creates an `outputs` folder containing CSV reports and a visualization.

## Concepts Used

### 1. Covariance by Matrix Multiplication

For a data matrix `X`, the centered matrix is:

```text
X_centered = X - mean(X)
```

The sample covariance matrix is calculated as:

```text
Covariance = X_centered @ X_centered.T / (n - 1)
```

The result is verified using:

```python
np.cov(data)
```

### 2. Markov Chains

A Markov chain represents transitions between states. This project uses:

- Sunny
- Cloudy
- Rainy

The steady-state probability is obtained from the eigenvector corresponding to eigenvalue 1 of the transposed transition matrix.

### 3. Time-Series Windowing

The project calculates:

- 7-day rolling mean
- 30-observation expanding standard deviation

NaN values produced at the beginning of the expanding calculation are handled using backward filling.

### 4. Visualization

The figure contains two subplots:

1. Scatter plot highlighting points outside two standard deviations from the mean.
2. Line plot showing raw values, rolling mean, and a shaded standard-deviation region.

## Output Files

```text
outputs/
├── simulated_variables.csv
├── manual_covariance_matrix.csv
├── numpy_covariance_matrix.csv
├── correlation_matrix.csv
├── steady_state_probabilities.csv
├── markov_chain_sequence.csv
├── time_series_analysis.csv
└── advanced_math_python_visualization.png
```
