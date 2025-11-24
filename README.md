# CCPP Energy Output Prediction

This project implements a machine learning pipeline to predict the electrical energy output (PE) of a Combined Cycle Power Plant.

## Project Structure
- `data/`: Contains the dataset (`CCPP_data.csv`).
- `src/`: Source code for the project.
    - `data_loader.py`: Handles data loading and splitting.
    - `model_trainer.py`: Trains models and evaluates performance.
    - `main.py`: Main entry point.
- `requirements.txt`: Python dependencies.

## Setup & Usage

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   ```bash
   python src/main.py
   ```

## Modeling Approach
- **Target**: Net hourly electrical energy output (PE).
- **Features**: Temperature (T), Ambient Pressure (AP), Relative Humidity (RH), Exhaust Vacuum (V).
- **Models Compared**:
    - Linear Regression (Baseline)
    - Random Forest Regressor (Complex)
- **Evaluation Metric**: Root Mean Squared Error (RMSE) and R² Score.

## Results
The model with the lowest RMSE on the validation set is selected as the final model and evaluated on the held-out test set.
