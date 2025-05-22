import numpy as np
import re
import matplotlib.pyplot as plt
import niceplots
plt.style.use(niceplots.get_style())
plt.rcParams["font.family"] = "Times New Roman"

def parse_zones_by_frequency(filename):
    with open(filename, 'r') as f:
        all_lines = f.readlines()

    # Stop reading at first occurrence of "enveloped"
    for i, line in enumerate(all_lines):
        if 'enveloped' in line.lower():
            all_lines = all_lines[:i]
            break

    data_dict = {}
    current_freq = None
    current_data = []

    freq_pattern = re.compile(r'(\d+\.\d+)hz')

    # Skip the first 7 lines (title, variables, etc.)
    all_lines = all_lines[7:]

    for line in all_lines:
        line = line.strip()

        if line.startswith('zone'):
            if current_freq is not None and current_data:
                data_array = np.array(current_data, dtype=np.float64)
                if current_freq in data_dict:
                    data_dict[current_freq].append(data_array)
                else:
                    data_dict[current_freq] = [data_array]
                current_data = []

            match = freq_pattern.search(line)
            if match:
                current_freq = float(match.group(1))

        elif line:
            try:
                values = [float(x) for x in line.split()]
                if len(values) == 5:
                    current_data.append(values)
            except ValueError:
                continue

    if current_freq is not None and current_data:
        data_array = np.array(current_data, dtype=np.float64)
        if current_freq in data_dict:
            data_dict[current_freq].append(data_array)
        else:
            data_dict[current_freq] = [data_array]

    for freq in data_dict:
        data_dict[freq] = np.vstack(data_dict[freq])

    return data_dict


# Usage
filename = 'nfactor_ts.dat'  # Replace with your actual file path
zone_data = parse_zones_by_frequency(filename)



# Plotting

from cycler import cycler
colors = plt.cm.tab20.colors  # or try 'tab20b', 'tab20c', etc.
#plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

plt.figure(figsize=(7, 5))

for i, (freq, data) in enumerate(sorted(zone_data.items())):
    if i >= 9:
        break
    x_c = data[:, 0]
    n_factor = data[:, 4]
    plt.plot(x_c, n_factor, label=f"{freq:.1f} Hz")

plt.xlabel("x/c", fontsize=16)
plt.ylabel("$N_{TS}$", fontsize=16, rotation=0, labelpad=20)
plt.xlim([0,0.6])
plt.ylim([-0.5,10])
plt.legend(
    title="Frequency",
    title_fontsize=16,
    fontsize=12,
    loc="center left",
    bbox_to_anchor=(0.6, 0.8),
    borderaxespad=0,
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust plot area to make space for legend
plt.tick_params(axis='both', labelsize=16)

#plt.show()
plt.savefig("nfactor_plot.svg", format='svg', dpi=300)