import tkinter as tk
from tkinter import messagebox
import pandas as pd
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/franztac/Water--End-to-End-ML_Proj.mlflow")
model_name = "Best Model"

# Attempt to load the production model
try:
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if versions:
        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/{model_name}"
        loaded_model = mlflow.pyfunc.load_model(logged_model)
    else:
        raise Exception("No model in Production stage.")

except Exception as e:
    loaded_model = None
    print(f"Error loading model: {e}")


# GUI Functionality
def make_prediction():
    if not loaded_model:
        messagebox.showerror("Error", "Model is not loaded.")
        return

    try:
        # Gather input data
        input_data = {
            "ph": float(entry_ph.get()),
            "Hardness": float(entry_hardness.get()),
            "Solids": float(entry_solids.get()),
            "Chloramines": float(entry_chloramines.get()),
            "Sulfate": float(entry_sulfate.get()),
            "Conductivity": float(entry_conductivity.get()),
            "Organic_carbon": float(entry_organic.get()),
            "Trihalomethanes": float(entry_trihalo.get()),
            "Turbidity": float(entry_turbidity.get()),
        }

        df = pd.DataFrame([input_data])
        prediction = loaded_model.predict(df)[0]
        result_label.config(text=f"Prediction: {prediction:.4f}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


# Create tkinter window
root = tk.Tk()
root.title("Water Quality Predictor")

# Define input fields
fields = {
    "ph": None,
    "Hardness": None,
    "Solids": None,
    "Chloramines": None,
    "Sulfate": None,
    "Conductivity": None,
    "Organic_carbon": None,
    "Trihalomethanes": None,
    "Turbidity": None,
}

entries = {}

for idx, field in enumerate(fields):
    tk.Label(root, text=field).grid(row=idx, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[field] = entry

entry_ph = entries["ph"]
entry_hardness = entries["Hardness"]
entry_solids = entries["Solids"]
entry_chloramines = entries["Chloramines"]
entry_sulfate = entries["Sulfate"]
entry_conductivity = entries["Conductivity"]
entry_organic = entries["Organic_carbon"]
entry_trihalo = entries["Trihalomethanes"]
entry_turbidity = entries["Turbidity"]

# Predict button
predict_button = tk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=10)

# Result label
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12, "bold"))
result_label.grid(row=len(fields) + 1, column=0, columnspan=2, pady=10)

root.mainloop()
