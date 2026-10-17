# ChurnSense
```bash
pip install -r requirements.txt
python train.py            # trains model -> models/
streamlit run app.py       # dashboard
uvicorn api:app --reload   # API docs at http://127.0.0.1:8000/docs
```
Retrain with `python train.py` whenever `data/telco_churn.csv` changes.
