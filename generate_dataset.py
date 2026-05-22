import pandas as pd
import random

data = []

for i in range(150):

    temp_error = round(random.uniform(0, 2), 2)
    humidity_error = random.randint(0, 20)
    oxygen_error = random.randint(0, 10)
    noise_level = random.randint(35, 70)

    device_age = random.randint(0, 10)
    last_maintenance_days = random.randint(0, 180)
    repair_history = random.randint(0, 5)

    # QC Rules
    if (
        temp_error <= 0.3
        and humidity_error <= 5
        and oxygen_error <= 2
        and noise_level <= 45
        and device_age <= 3
        and last_maintenance_days <= 60
        and repair_history <= 1
    ):
        qc_status = 0

    elif (
        temp_error <= 0.7
        and humidity_error <= 10
        and oxygen_error <= 5
        and noise_level <= 55
    ):
        qc_status = 1

    else:
        qc_status = 2

    data.append([
        temp_error,
        humidity_error,
        oxygen_error,
        noise_level,
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
    "device_age",
    "last_maintenance_days",
    "repair_history",
    "qc_status"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/incubator_qc_dataset.csv", index=False)

print("Dataset created successfully!")