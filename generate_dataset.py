import pandas as pd
import random

data = []

for i in range(150):
    temp_error = round(random.uniform(0, 2), 2)
    humidity_error = random.randint(0, 20)
    oxygen_error = random.randint(0, 10)
    noise_level = random.randint(35, 70)
    airflow = round(random.uniform(0.05, 0.138), 3)  # measured airflow in m/s

    device_age = random.randint(0, 10)
    last_maintenance_days = random.randint(0, 180)
    repair_history = random.randint(0, 5)

    # QC Rules (simplified scoring for dataset)
    if (
        temp_error <= 0.3
        and humidity_error <= 5
        and oxygen_error <= 2
        and noise_level <= 45
        and 0.05 <= airflow <= 0.138
        and device_age <= 3
        and last_maintenance_days <= 60
        and repair_history <= 1
    ):
        qc_status = 0  # Normal
    elif (
        temp_error <= 0.7
        and humidity_error <= 10
        and oxygen_error <= 5
        and noise_level <= 55
        and 0.05 <= airflow <= 0.138
    ):
        qc_status = 1  # Warning
    else:
        qc_status = 2  # Failure

    data.append([
        temp_error,
        humidity_error,
        oxygen_error,
        noise_level,
        airflow,
        device_age,
        last_maintenance_days,
        repair_history,
        qc_status
    ])

columns = [
    "temp_error",
    "humidity_error",
    "oxygen_error",
    "noise_level",
    "airflow",
    "device_age",
    "last_maintenance_days",
    "repair_history",
    "qc_status"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("data/incubator_qc_dataset.csv", index=False)
