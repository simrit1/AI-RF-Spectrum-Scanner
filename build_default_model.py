from pathlib import Path
from rf_core import train_default_model
p = Path(__file__).resolve().parent / "models" / "house_rf_model.joblib"
print(train_default_model(str(p)))
