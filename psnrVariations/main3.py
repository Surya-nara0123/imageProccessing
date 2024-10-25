import random
import numpy as np
import cv2
import time

# Logistic Map Chaotic Sequence Generator
def logistic_map_keygen(x0, length, r=3.99):
    key = []
    x = x0
    for _ in range(length):
        x = r * x * (1 - x)
        key.append(int(x * 255))  # Scale the value to 0-255 range for byte-sized key
    return key

# Function to compute psnr between encrypted image and original image
def calculate_psnr(original_image, compressed_image):
    # Ensure both images have the same shape
    if original_image.shape != compressed_image.shape:
        raise ValueError("The images must have the same dimensions.")

    mse = np.mean((original_image - compressed_image) ** 2)
    
    if mse == 0:
        return float('inf')  # If MSE is 0, PSNR is infinite (perfect match)

    max_pixel_value = 255.0  # Assuming 8-bit grayscale image
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# Emperor Penguin Optimizer for chaotic key optimization
def epo_optimize(img, population_size, iterations, chaotic_sequence_length):
    height, width = img.shape
    best_initial_value = None
    best_psnr = float('inf')

    # Initialize penguin population (initial values for chaotic maps)
    population = [random.random() for _ in range(population_size)]

    avg_psnr_values = []
    for _ in range(iterations):
        # Track the best psnr and corresponding initial value in the population
        psnr_values = []
        for initial_value in population:
            encrypted_img1 = np.zeros((height, width), dtype=object)
            chaotic_key = logistic_map_keygen(initial_value, height * width)

            # Encrypt the image using XOR with the chaotic key
            k = 0  # Index for the chaotic key
            for i in range(height):
                for j in range(width):
                    encrypted_img1[i, j] = img[i, j] ^ chaotic_key[k]  # XOR operation
                    k += 1

            # Calculate psnr for the current chaotic key
            psnr = calculate_psnr(encrypted_img1, img)

            # Update the best initial value if this one has a higher psnr
            if psnr < best_psnr:
                best_psnr = psnr
                best_initial_value = initial_value
                
            print("#" * 50)
            print("Best initial value: ", best_initial_value)
            print("Best psnr: ", best_psnr)
            print("#" * 50)
            psnr_values.append(best_psnr)
        psnr_values = set(psnr_values)
        avg_psnr_values.append(sum(psnr_values) / len(psnr_values))

        # Move penguins (keys) based on the best-performing solution
        for i in range(population_size):
            if random.random() < 0.3:  # Introduce random movement for diversity
                population[i] = random.random()
            else:
                # Move towards the best initial value by adding a small random tweak
                tweak = random.uniform(-0.01, 0.01)
                population[i] = min(max(population[i] + tweak, 0), 1)  # Keep it in [0, 1] range
    # Plot the average psnr values over iterations
    # import matplotlib.pyplot as plt
    # plt.plot(avg_psnr_values)
    # plt.xlabel("Iterations")
    # plt.ylabel("Average psnr")
    # plt.title("Average psnr values over iterations")
    # plt.show()
    
    # save the values in a file
    with open(f"psnr_values_logistic_maps_{population_size}_{iterations}.txt", "w") as f:
        for item in avg_psnr_values:
            f.write("%s\n" % item)
    return best_initial_value, best_psnr


# Chaotic Map-based Encryption function (using XOR)
def chaotic_encrypt(img, chaotic_key):
    height, width = img.shape
    encrypted_img = np.zeros((height, width), dtype=np.uint8)
    k = 0  # Index for the chaotic key
    for i in range(height):
        for j in range(width):
            encrypted_img[i, j] = img[i, j] ^ chaotic_key[k]  # XOR each pixel with chaotic key
            k += 1
    return encrypted_img

# Chaotic Map-based Decryption function (same as encryption since XOR is symmetric)
def chaotic_decrypt(encrypted_img, chaotic_key):
    return chaotic_encrypt(encrypted_img, chaotic_key)  # XOR again to decrypt

def main():
    img = cv2.imread("../../img.png", 0)
    height, width = img.shape

    currTime = time.time()

    # Emperor Penguin Optimization with chaotic key generation
    chaotic_sequence_length = height * width
    for i in range(3):
        for j in range(3):
            best_initial_value, best_psnr = epo_optimize(img, population_size=10*(i+1), iterations=50*(j+1), chaotic_sequence_length=chaotic_sequence_length)
            print(f"Best initial value for chaotic map: {best_initial_value}, Best psnr: {best_psnr}")
            print("Time taken: ", time.time() - currTime)


if __name__ == "__main__":
    main()
