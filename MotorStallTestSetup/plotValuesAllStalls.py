import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 1. Define the folder path dynamically
folder_path = Path("stall_motor_tests")

# 2. Grab all CSV files inside that folder sequentially
# .glob('*.csv') finds everything ending in .csv inside the folder
csv_files = sorted(folder_path.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in directory: {folder_path.resolve()}")

# 3. Loop through every single file one by one
for filename in csv_files:
    print(f"Processing and plotting: {filename.name}...")

    x_data, y_data = [], []

    # Open the current file in the loop iteration
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # skip header row
        except StopIteration:
            continue  # skip if file is completely empty

        for row in reader:
            try:
                t = float(row[0])  # time (µs or ms)
                current = float(row[1]) * 1000  # convert to mA

                x_data.append(t)
                y_data.append(current)
            except (ValueError, IndexError):
                pass  # skip corrupted or malformed rows cleanly

    # Skip files that didn't yield any valid numeric rows
    if len(x_data) == 0:
        print(f"  Skipping {filename.name}: No valid numeric data found.")
        continue

    # --- Normalize time ---
    x_data = np.array(x_data)
    x_data = (x_data - x_data[0]) / 1000.0  # µs → ms

    y_data = np.array(y_data)

    # --- Downsample ---
    DOWNSAMPLE = 10
    x_data = x_data[::DOWNSAMPLE]
    y_data = y_data[::DOWNSAMPLE]

    # --- Smooth signal (moving average) ---
    WINDOW = 5
    if len(y_data) >= WINDOW:
        y_smooth = np.convolve(y_data, np.ones(WINDOW) / WINDOW, mode="same")
    else:
        y_smooth = y_data  # fallback if data is shorter than the filter window

    # --- Create Isolated Plot Configuration ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_data, y_smooth, linewidth=1, color="crimson")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (mA)")

    # Dynamically name the title based on which file is currently open
    ax.set_title(f"Motor Current vs Time ({filename.name})")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))

    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save each plot uniquely (e.g., 'plot_motor_data1.png', 'plot_motor_data2.png')
    output_image_name = f"plot_{filename.stem}.png"
    plt.savefig(output_image_name)

    # Display the current single plot window on screen
    plt.show()

    # CRITICAL: Close the figure instance to clear RAM memory before opening the next file
    plt.close(fig)

print("\nFinished generating all separate motor charts successfully!")