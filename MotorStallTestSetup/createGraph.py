import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("rpm_400.csv")

plt.figure(figsize=(12,5))
plt.plot(df["elapsed_time"], df["current_mA"])
plt.xlabel("Time")
plt.ylabel("Current (mA)")
plt.title("Motor Current vs Time")
plt.grid()
plt.show()

df = pd.read_csv("rpm_400.csv")

plt.figure(figsize=(12,5))
plt.plot(df["elapsed_time"], df["rpm_command"])
plt.xlabel("Time")
plt.ylabel("RPM Command")
plt.title("RPM Command vs Time")
plt.grid()
plt.show()