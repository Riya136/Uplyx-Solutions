# Sequential Data Analysis: RNN vs LSTM vs GRU

This project performs sentiment classification on the **Women's E-Commerce Clothing Reviews** dataset using three recurrent neural network architectures:

- Standard RNN
- LSTM
- GRU

The project follows the internship assignment requirements.

## Features

- Text cleaning and tokenization
- Vocabulary creation
- Conversion of words into vocabulary indices
- Sequence padding
- Word embeddings
- Sentiment classification
- RNN, LSTM, and GRU model training
- Validation accuracy comparison
- Training-time comparison
- Trainable-parameter comparison
- Accuracy and loss plots
- Classification reports
- Saved model weights

## Dataset

Recommended dataset:

**Women's E-Commerce Clothing Reviews** from Kaggle.

The CSV should contain columns similar to:

```text
Review Text
Recommended IND
```

The code automatically detects common variations of these column names.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py --csv_path "Womens Clothing E-Commerce Reviews.csv" --epochs 5
```

Example:

```bash
python main.py --csv_path "D:/datasets/Womens Clothing E-Commerce Reviews.csv" --epochs 5
```

## Output Files

The program creates an `outputs` folder:

```text
outputs/
├── rnn_accuracy.png
├── rnn_loss.png
├── lstm_accuracy.png
├── lstm_loss.png
├── gru_accuracy.png
├── gru_loss.png
├── rnn_weights.pth
├── lstm_weights.pth
├── gru_weights.pth
├── rnn_classification_report.csv
├── lstm_classification_report.csv
├── gru_classification_report.csv
├── rnn_lstm_gru_comparison.csv
└── vocabulary_size.txt
```

## Architecture Explanation

### RNN

A standard RNN passes a hidden state from one time step to the next. It is simple but can suffer from the vanishing-gradient problem when learning long-term dependencies.

### LSTM

An LSTM contains:

- Cell state
- Forget gate
- Input gate
- Output gate

The cell state works like a continuous information conveyor belt. The gates control what information should be removed, added, and exposed. This helps LSTM preserve information over longer sequences.

### GRU

A GRU uses:

- Update gate
- Reset gate

It has fewer gates than LSTM and does not maintain a separate cell state. It is usually simpler and faster while still helping with long-term dependencies.

## Comparison

The program measures:

- Training time in seconds
- Final validation accuracy
- Total trainable parameters

The actual values depend on the dataset, number of epochs, batch size, and hardware.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
