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

# Function to compute UACI between encrypted image and original image
def calculate_uaci(encrypted_img1, img):
    height, width = img.shape
    uaci = 0
    for i in range(height):
        for j in range(width):
            uaci += abs(encrypted_img1[i, j] - img[i, j])
    uaci /= height * width * 255 / 100
    return uaci

# Emperor Penguin Optimizer for chaotic key optimization
def epo_optimize(img, population_size, iterations, chaotic_sequence_length):
    height, width = img.shape
    best_initial_value = None
    best_uaci = -float('inf')

    # Initialize penguin population (initial values for chaotic maps)
    population = [random.random() for _ in range(population_size)]

    avg_uaci_values = []
    for _ in range(iterations):
        # Track the best UACI and corresponding initial value in the population
        uaci_values = []
        for initial_value in population:
            encrypted_img1 = np.zeros((height, width), dtype=object)
            chaotic_key = logistic_map_keygen(initial_value, height * width)

            # Encrypt the image using XOR with the chaotic key
            k = 0  # Index for the chaotic key
            for i in range(height):
                for j in range(width):
                    encrypted_img1[i, j] = img[i, j] ^ chaotic_key[k]  # XOR operation
                    k += 1

            # Calculate UACI for the current chaotic key
            uaci = calculate_uaci(encrypted_img1, img)

            # Update the best initial value if this one has a higher UACI
            if uaci > best_uaci:
                best_uaci = uaci
                best_initial_value = initial_value
                
            print("#" * 50)
            print("Best initial value: ", best_initial_value)
            print("Best UACI: ", best_uaci)
            print("#" * 50)
            uaci_values.append(best_uaci)
        uaci_values = set(uaci_values)
        avg_uaci_values.append(sum(uaci_values) / len(uaci_values))

        # Move penguins (keys) based on the best-performing solution
        for i in range(population_size):
            if random.random() < 0.3:  # Introduce random movement for diversity
                population[i] = random.random()
            else:
                # Move towards the best initial value by adding a small random tweak
                tweak = random.uniform(-0.01, 0.01)
                population[i] = min(max(population[i] + tweak, 0), 1)  # Keep it in [0, 1] range
    # Plot the average UACI values over iterations
    # import matplotlib.pyplot as plt
    # plt.plot(avg_uaci_values)
    # plt.xlabel("Iterations")
    # plt.ylabel("Average UACI")
    # plt.title("Average UACI values over iterations")
    # plt.show()
    
    # save the values in a file
    with open(f"iteration1/uaci_values_logistic_maps_{population_size}_{iterations}.txt", "w") as f:
        for item in avg_uaci_values:
            f.write("%s\n" % item)
    return best_initial_value, best_uaci


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
    img = cv2.imread("../img.png", 0)
    height, width = img.shape

    currTime = time.time()

    # Emperor Penguin Optimization with chaotic key generation
    chaotic_sequence_length = height * width
    # for i in range(3):
        # for j in range(3):
            # if (i == 0 and j != 1) or (i == 1 and j != 0) or (i == 2 and j != 0):
    best_initial_value, best_uaci = epo_optimize(img, population_size=10*(1), iterations=50*(1), chaotic_sequence_length=chaotic_sequence_length)
    print(f"Best initial value for chaotic map: {best_initial_value}, Best UACI: {best_uaci}")

    # Generate the chaotic key using the best initial value
    best_chaotic_key = logistic_map_keygen(best_initial_value, chaotic_sequence_length)

    # Encrypt the image using the best chaotic key
    encrypted_img = chaotic_encrypt(img, best_chaotic_key)

    # Decrypt the image
    decrypted_img = chaotic_decrypt(encrypted_img, best_chaotic_key)

    print("Time taken: ", time.time() - currTime)

    # Compare original and decrypted images
    if np.array_equal(img, decrypted_img):
        print("Success: The decrypted image is identical to the original image.")
    else:
        print("Error: The decrypted image differs from the original image.")

    # Display images
    cv2.imshow("Original Image", img)
    cv2.imshow("Encrypted Image", encrypted_img)
    cv2.imshow("Decrypted Image", decrypted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
