import random
import numpy as np
import cv2
import time


# Henon Map Chaotic Sequence Generator
def henon_map_keygen(x0, y0, length, a=1.4, b=0.3):
    x = x0
    y = y0
    key = []
    for _ in range(length):
        x_new = 1 - a * x * (x%256) + y %256
        y_new = b * x
        x, y = x_new, y_new
        key.append(
            int(abs(x * 255))%256
        )  # Normalize to 0-255 range and mod 256 for byte-size keys
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


# Emperor Penguin Optimizer for chaotic key optimization (Henon Map version)
def epo_optimize(img, population_size, iterations, chaotic_sequence_length):
    height, width = img.shape
    best_initial_values = None
    best_uaci = -float("inf")

    # Initialize penguin population (initial values for Henon maps)
    population = [(random.random(), random.random()) for _ in range(population_size)]

    avg_uaci_values = []
    for _ in range(iterations):
        # Track the best UACI and corresponding initial values in the population
        uaci_values = []
        for initial_values in population:
            x0, y0 = initial_values
            encrypted_img1 = np.zeros((height, width), dtype=object)
            chaotic_key = henon_map_keygen(x0, y0, height * width)

            # Encrypt the image using XOR with the chaotic key
            k = 0  # Index for the chaotic key
            for i in range(height):
                for j in range(width):
                    encrypted_img1[i, j] = img[i, j] ^ chaotic_key[k]  # XOR operation
                    k += 1

            # Calculate UACI for the current chaotic key
            uaci = calculate_uaci(encrypted_img1, img)

            # Update the best initial values if this one has a higher UACI
            if uaci > best_uaci:
                best_uaci = uaci
                best_initial_values = initial_values

            print("#" * 50)
            print("Best initial value: ", best_initial_values)
            print("Best UACI: ", best_uaci)
            print("#" * 50)
            uaci_values.append(best_uaci)
        uaci_values = set(uaci_values)
        avg_uaci_values.append(sum(uaci_values) / len(uaci_values))

        # Move penguins (keys) based on the best-performing solution
        for i in range(population_size):
            if random.random() < 0.3:  # Introduce random movement for diversity
                population[i] = (random.random(), random.random())
            else:
                # Move towards the best initial values by adding a small random tweak
                tweak_x = random.uniform(-0.01, 0.01)
                tweak_y = random.uniform(-0.01, 0.01)
                new_x = min(max(population[i][0] + tweak_x, 0), 1)
                new_y = min(max(population[i][1] + tweak_y, 0), 1)
                population[i] = (new_x, new_y)

    # save the values in a file
    with open("uaci_values_henon_20.txt", "w") as f:
        for item in avg_uaci_values:
            f.write("%s\n" % item)
    return best_initial_values, best_uaci


# Chaotic Map-based Encryption function (using XOR)
def chaotic_encrypt(img, chaotic_key):
    height, width = img.shape
    encrypted_img = np.zeros((height, width), dtype=np.uint8)
    k = 0  # Index for the chaotic key
    for i in range(height):
        for j in range(width):
            encrypted_img[i, j] = (
                img[i, j] ^ chaotic_key[k]
            )  # XOR each pixel with chaotic key
            k += 1
    return encrypted_img


# Chaotic Map-based Decryption function (same as encryption since XOR is symmetric)
def chaotic_decrypt(encrypted_img, chaotic_key):
    return chaotic_encrypt(encrypted_img, chaotic_key)  # XOR again to decrypt


def main():
    img = cv2.imread("../img.png", 0)  # Load the image in grayscale
    height, width = img.shape

    currTime = time.time()

    # Emperor Penguin Optimization with chaotic key generation (Henon Map)
    chaotic_sequence_length = height * width
    best_initial_values, best_uaci = epo_optimize(
        img,
        population_size=10,
        iterations=200,
        chaotic_sequence_length=chaotic_sequence_length,
    )
    print(
        f"Best initial values for Henon map: {best_initial_values}, Best UACI: {best_uaci}"
    )

    # Generate the chaotic key using the best initial values
    best_chaotic_key = henon_map_keygen(
        best_initial_values[0], best_initial_values[1], chaotic_sequence_length
    )

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