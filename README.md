## TripFare: Predicting Urban Taxi Fare with Machine Learning

### Quickstart

- Create and activate a virtual environment
- Install dependencies
- Train model
- Run Streamlit app

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python train.py --data-path taxi_fare.csv --sample 150000
streamlit run app.py
```

### Project Structure

- `taxi_fare.csv`: source dataset
- `src/`: data loading, features, preprocessing, modeling utilities
- `artifacts/`: trained model and metrics
- `notebooks/`: EDA
- `train.py`: end-to-end training
- `app.py`: Streamlit UI

### Notes
- Target variable: `total_amount`
- Engineered features include haversine distance, pickup hour/day, weekend, am/pm, night flag, and trip duration.
- Best model is saved to `artifacts/best_model.pkl` and loaded by the app. 