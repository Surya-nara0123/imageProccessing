import matplotlib.pyplot as plt
import os


# Function to read numbers from a file
def read_numbers_from_file(filename):
    with open(filename, "r") as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return numbers


# Read numbers from file
filename = "iteration1/uaci_values_henon.txt"
filename1 = "iteration1/uaci_values_henon_20.txt"
filename2 = "iteration1/uaci_values_henon_30.txt"
filename3 = "iteration1/uaci_values_henon_10_100.txt"
filename4 = "iteration1/uaci_values_henon_10_150.txt"
filename5 = "iteration1/uaci_values_henon_10_200.txt"
# numbers = read_numbers_from_file(filename)
# numbers1 = read_numbers_from_file(filename1)
# numbers2 = read_numbers_from_file(filename2)
# numbers3 = read_numbers_from_file(filename3)
# numbers4 = read_numbers_from_file(filename4)
# numbers5 = read_numbers_from_file(filename5)[:-2:]

# # Generate x-values (just indices for each number)
# x_values = range(len(numbers))
# x_values1 = range(len(numbers1))
# x_values2 = range(len(numbers2))
# x_values3 = range(len(numbers3))
# x_values4 = range(len(numbers4))
# x_values5 = range(len(numbers5))

# # Plot the numbers
# plt.plot(x_values, numbers, linestyle='-', color='g')
# # plt.plot(x_values1, numbers1, linestyle='-', color='r')
# # plt.plot(x_values2, numbers2, linestyle='-', color='g')
# plt.plot(x_values3, numbers3, linestyle='-', color='b')
# plt.plot(x_values4, numbers4, linestyle='-', color='y')
# plt.plot(x_values5, numbers5, linestyle='-', color='r')

colors = [
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "orange",
]
# Read numbers from file
directory = "iteration1"
# print(len(os.listdir(directory)), len(os.listdir(directory)))
# for filename in os.listdir(directory):
#     if filename.endswith("_50.txt") and filename.startswith("psnr_values_henon"):
#         numbers = read_numbers_from_file(os.path.join(directory, filename))
#         x_values = range(len(numbers))
#         a = os.listdir(directory).index(filename)
#         plt.plot(x_values, numbers, linestyle="-", color=colors[a%6])
#     else:
#         continue
numbers = read_numbers_from_file(os.path.join(directory, "uaci_values_aes_10_50.txt"))
x_values = range(len(numbers))
plt.plot(x_values, numbers, linestyle="-", color='r')
numbers = read_numbers_from_file(os.path.join(directory, "uaci_values_aes_20_50.txt"))
x_values = range(len(numbers))
plt.plot(x_values, numbers, linestyle="-", color='b')
numbers = read_numbers_from_file(os.path.join(directory, "uaci_values_aes_30_50.txt"))
x_values = range(len(numbers))
plt.plot(x_values, numbers, linestyle="-", color='g')


# a red line over 33.3333
# plt.axhline(y=33.3333, color='r', linestyle='--')

# add a legend
plt.legend([f"{i} penguins" for i in range(10, 40, 10)])

# Add labels and title
plt.xlabel("iteration number")
plt.ylabel("best PSNR value")
plt.title("PSNR values of aes over different population sizes")

# Show the plot
plt.show()
