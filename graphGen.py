import matplotlib.pyplot as plt

# Function to read numbers from a file
def read_numbers_from_file(filename):
    with open(filename, 'r') as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return numbers

population_sizes = [10, 20, 30]
iterations = [50, 100, 150]

# Plot the UACI values for different population sizes
plt.figure(figsize=(12, 6))
for population_size in population_sizes:
    uaci_values = read_numbers_from_file(f"psnrVariations/psnr_values_lorenz_{population_size}_50.txt")
    plt.plot(uaci_values, label=f'Population size: {population_size}')
plt.xlabel("Iterations")
plt.ylabel("Average UACI")
plt.title("Average UACI values over iterations for different population sizes")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("psnr_values_lorenz_population_sizes.png")

# Plot the UACI values for different numbers of iterations
plt.figure(figsize=(12, 6))
for iteration in iterations:
    uaci_values = read_numbers_from_file(f"psnrVariations/psnr_values_lorenz_10_{iteration}.txt")
    plt.plot(uaci_values, label=f'Iterations: {iteration}')
plt.xlabel("Iterations")
plt.ylabel("Average UACI")
plt.title("Average UACI values over iterations for different numbers of iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("psnr_values_lorenz_iterations.png")

# Show the plot
# plt.show()
