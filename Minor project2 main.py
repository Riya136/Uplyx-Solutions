from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 42
OUTPUT_DIR = Path("outputs")


def covariance_matrix_manual(data):
    """Calculate covariance using matrix multiplication."""
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("Data must be a 2D matrix.")

    observations = data.shape[1]
    mean_vector = np.mean(data, axis=1, keepdims=True)
    centered_data = data - mean_vector

    covariance = (
        centered_data @ centered_data.T
    ) / (observations - 1)

    return covariance


def covariance_and_correlation_demo():
    """Generate variables and calculate covariance/correlation matrices."""
    rng = np.random.default_rng(SEED)

    data = rng.normal(
        loc=[50, 30, 70],
        scale=[10, 5, 15],
        size=(100, 3),
    ).T

    manual_covariance = covariance_matrix_manual(data)
    numpy_covariance = np.cov(data)

    correlation_matrix = np.corrcoef(data)

    print("\n" + "=" * 60)
    print("PART 1.1: COVARIANCE AND CORRELATION MATRICES")
    print("=" * 60)
    print("\nData shape:", data.shape)

    print("\nManual covariance matrix:")
    print(np.round(manual_covariance, 4))

    print("\nNumPy covariance matrix:")
    print(np.round(numpy_covariance, 4))

    print(
        "\nManual and NumPy covariance matrices are equal:",
        np.allclose(manual_covariance, numpy_covariance),
    )

    print("\nCorrelation matrix:")
    print(np.round(correlation_matrix, 4))

    return data, manual_covariance, numpy_covariance, correlation_matrix


def markov_chain_demo():
    """Simulate a 3-state Markov chain and calculate steady-state values."""
    states = ["Sunny", "Cloudy", "Rainy"]

    transition_matrix = np.array([
        [0.60, 0.30, 0.10],
        [0.20, 0.50, 0.30],
        [0.10, 0.30, 0.60],
    ])

    current_state = 0
    steps = 50
    rng = np.random.default_rng(SEED)

    state_sequence = [states[current_state]]

    for _ in range(steps):
        current_state = rng.choice(
            len(states),
            p=transition_matrix[current_state],
        )
        state_sequence.append(states[current_state])

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    steady_state_index = np.argmin(np.abs(eigenvalues - 1))
    steady_state_vector = np.real(
        eigenvectors[:, steady_state_index]
    )
    steady_state_vector = steady_state_vector / steady_state_vector.sum()

    print("\n" + "=" * 60)
    print("PART 1.2: MARKOV CHAIN")
    print("=" * 60)
    print("\nStates:", states)
    print("\nTransition matrix:")
    print(transition_matrix)

    print("\nSimulated state sequence:")
    print(" -> ".join(state_sequence))

    print("\nEigenvalues:")
    print(np.round(eigenvalues, 4))

    print("\nSteady-state probabilities:")
    for state, probability in zip(states, steady_state_vector):
        print(f"{state}: {probability:.4f}")

    return (
        states,
        transition_matrix,
        state_sequence,
        steady_state_vector,
    )


def time_series_demo():
    """Create daily data and calculate rolling/expanding statistics."""
    rng = np.random.default_rng(SEED)

    dates = pd.date_range(
        start="2025-01-01",
        periods=365,
        freq="D",
    )

    values = (
        50
        + 10 * np.sin(np.linspace(0, 8 * np.pi, 365))
        + rng.normal(0, 4, 365)
    )

    df = pd.DataFrame(
        {"Value": values},
        index=dates,
    )

    df["Rolling_Mean_7D"] = df["Value"].rolling(
        window=7,
        min_periods=1,
    ).mean()

    df["Expanding_Std_30D"] = df["Value"].expanding(
        min_periods=30,
    ).std()

    # Handle NaN values created during the first 29 observations.
    df["Expanding_Std_30D"] = df["Expanding_Std_30D"].bfill()

    print("\n" + "=" * 60)
    print("PART 2.1: TIME-SERIES WINDOWING")
    print("=" * 60)
    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values after handling NaN:")
    print(df.isna().sum())

    return df


def advanced_visualization(df, output_dir=OUTPUT_DIR):
    """Create the required two-subplot visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_value = df["Value"].mean()
    standard_deviation = df["Value"].std()

    lower_limit = mean_value - 2 * standard_deviation
    upper_limit = mean_value + 2 * standard_deviation

    outside_two_std = (
        (df["Value"] < lower_limit)
        | (df["Value"] > upper_limit)
    )

    dates = df.index
    values = df["Value"]
    rolling_mean = df["Rolling_Mean_7D"]
    rolling_std = df["Value"].rolling(
        window=7,
        min_periods=1,
    ).std().fillna(0)

    upper_band = rolling_mean + rolling_std
    lower_band = rolling_mean - rolling_std

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
    )

    axes[0].scatter(
        dates[~outside_two_std],
        values[~outside_two_std],
        label="Within 2 standard deviations",
        alpha=0.7,
    )
    axes[0].scatter(
        dates[outside_two_std],
        values[outside_two_std],
        label="Outside 2 standard deviations",
        alpha=0.9,
    )
    axes[0].axhline(
        mean_value,
        linestyle="--",
        label="Mean",
    )
    axes[0].set_title("Daily Data and Outliers")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(
        dates,
        values,
        label="Raw daily data",
        alpha=0.5,
    )
    axes[1].plot(
        dates,
        rolling_mean,
        label="7-day rolling mean",
        linewidth=2,
    )
    axes[1].fill_between(
        dates,
        lower_band,
        upper_band,
        alpha=0.2,
        label="Rolling mean ± standard deviation",
    )
    axes[1].set_title("Raw Data with Rolling Mean")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Advanced Applications in Math and Python")
    fig.tight_layout()

    figure_path = output_dir / "advanced_math_python_visualization.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisualization saved to: {figure_path.resolve()}")

    return figure_path


def save_outputs(data, manual_covariance, numpy_covariance,
                 correlation_matrix, states, transition_matrix,
                 state_sequence, steady_state_vector, df):
    """Save numerical outputs and generated data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(data.T, columns=[
        "Variable_1", "Variable_2", "Variable_3"
    ]).to_csv(OUTPUT_DIR / "simulated_variables.csv", index=False)

    pd.DataFrame(
        manual_covariance,
        index=["Variable_1", "Variable_2", "Variable_3"],
        columns=["Variable_1", "Variable_2", "Variable_3"],
    ).to_csv(OUTPUT_DIR / "manual_covariance_matrix.csv")

    pd.DataFrame(
        numpy_covariance,
        index=["Variable_1", "Variable_2", "Variable_3"],
        columns=["Variable_1", "Variable_2", "Variable_3"],
    ).to_csv(OUTPUT_DIR / "numpy_covariance_matrix.csv")

    pd.DataFrame(
        correlation_matrix,
        index=["Variable_1", "Variable_2", "Variable_3"],
        columns=["Variable_1", "Variable_2", "Variable_3"],
    ).to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    pd.DataFrame({
        "State": states,
        "Steady_State_Probability": steady_state_vector,
    }).to_csv(OUTPUT_DIR / "steady_state_probabilities.csv", index=False)

    pd.DataFrame({
        "Step": range(len(state_sequence)),
        "State": state_sequence,
    }).to_csv(OUTPUT_DIR / "markov_chain_sequence.csv", index=False)

    df.to_csv(OUTPUT_DIR / "time_series_analysis.csv")

    print(f"All reports saved in: {OUTPUT_DIR.resolve()}")


def main():
    data, manual_covariance, numpy_covariance, correlation_matrix = (
        covariance_and_correlation_demo()
    )

    states, transition_matrix, state_sequence, steady_state_vector = (
        markov_chain_demo()
    )

    df = time_series_demo()
    advanced_visualization(df)

    save_outputs(
        data,
        manual_covariance,
        numpy_covariance,
        correlation_matrix,
        states,
        transition_matrix,
        state_sequence,
        steady_state_vector,
        df,
    )

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
