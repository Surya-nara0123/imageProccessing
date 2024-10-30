import matplotlib.pyplot as plt
import os


# Function to read numbers from a file
def read_numbers_from_file(filename):
    with open(filename, "r") as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return numbers


colors = [
    "red",
    "blue",
    "green",
    "violet",
    "purple",
    "orange",
]
# Read numbers from file
directory = "iteration1"

population_sizes = [10, 20, 30]
iterations = [50, 100, 150]
i = 10;
for j in iterations:
    numbers = read_numbers_from_file(os.path.join(directory, f"uaci_values_logistic_maps_{i}_{j}.txt"))
    x_values = range(len(numbers))
    a = population_sizes.index(i)
    b = iterations.index(j)
    plt.plot(x_values, numbers, linestyle="-", color=colors[(a+b)%3])
plt.legend([f"{i} iterations" for i in iterations])
# Add labels and title
plt.xlabel("iteration number")
plt.ylabel("best UACI value")
plt.title("UACI values of logestic maps over different iterations")
plt.savefig(f"iteration1/UACI logestic maps population {i} iteration {iterations}.png")
