import random
from math import pow
import numpy as np
import cv2
import time


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Random key generator
def gen_key(q):
    key = random.randint(pow(10, 50), q)  # Increase key range
    while gcd(q, key) != 1:
        key = random.randint(pow(10, 50), q)
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

# Emperor Penguin Optimizer for key generation
def epo_optimize(img, population_size, iterations, q, g, h):
    height, width = img.shape
    best_key = None
    best_uaci = -float('inf')

    # Initialize penguin population (keys)
    population = [gen_key(q) for _ in range(population_size)]

    for _ in range(iterations):
        # Track the best UACI and corresponding key in the population
        for key in population:
            encrypted_img1 = np.zeros((height, width), dtype=object)
            for i in range(height):
                for j in range(width):
                    msg = chr(img[i][j])
                    en_msg, p = encrypt(msg, q, h, g)
                    encrypted_img1[i][j] = en_msg[0] % 256  # Store the first encrypted byte

            # Calculate UACI for the current key
            uaci = calculate_uaci(encrypted_img1, img)

            # Update the best key if this one has a higher UACI
            if uaci > best_uaci:
                best_uaci = uaci
                best_key = key
                
            print("#" * 50)
            print("Best key: ", best_key)
            print("Best UACI: ", best_uaci)
            print("#" * 50)

        # Move penguins (keys) based on the best-performing solution
        for i in range(population_size):
            if random.random() < 0.3:  # Introduce random movement for diversity
                population[i] = gen_key(q)
            else:
                # Move towards the best key by adding a small random tweak
                tweak = random.randint(-1000, 1000)
                population[i] = (best_key + tweak) % q
    return best_key, best_uaci


def power(a, b, c):
    x = 1
    y = a
    while b > 0:
        if b % 2 != 0:
            x = (x * y) % c
        y = (y * y) % c
        b //= 2
    return x % c


def encrypt(msg, q, h, g):
    en_msg = []
    k = gen_key(q)
    s = power(h, k, q)
    p = power(g, k, q)
    for char in msg:
        en_msg.append(s * ord(char) + random.randint(1, 256))  # Introduce randomness
    return en_msg, p


def decrypt(en_msg, p, key, q):
    dr_msg = []
    h = power(p, key, q)
    for num in en_msg:
        decrypted_val = int((num - random.randint(1, 256)) / h)
        if 0 <= decrypted_val < 0x110000:  # Ensure value is in valid range
            dr_msg.append(chr(decrypted_val))
        else:
            dr_msg.append("?")  # Use a placeholder for invalid values
    return dr_msg


def main():
    img = cv2.imread("../img.png", 0)
    height, width = img.shape
    encrypted_img = np.zeros((height, width), dtype=object)
    encrypted_img1 = np.zeros((height, width), dtype=object)
    decrypted_img = np.zeros((height, width), dtype=np.uint8)

    currTime = time.time()
    q = random.randint(pow(10, 50), pow(10, 70))
    g = random.randint(2, q)
    key = gen_key(q)
    h = power(g, key, q)

    # Emperor Penguin Optimization
    best_key, best_uaci = epo_optimize(img, population_size=10, iterations=50, q=q, g=g, h=h)
    print(f"Best key: {best_key}, Best UACI: {best_uaci}")

    for i in range(height):
        for j in range(width):
            msg = chr(img[i][j])

            en_msg, p = encrypt(msg, q, h, g)
            encrypted_img[i][j] = en_msg
            encrypted_img1[i][j] = en_msg[0] % 256

            dr_msg = decrypt(en_msg, p, best_key, q)
            decrypted_pixel_value = ord(dr_msg[0])
            decrypted_img[i, j] = decrypted_pixel_value

    print("Time taken: ", time.time() - currTime)

    # Compare original and decrypted images
    if np.array_equal(img, decrypted_img):
        print("Success: The decrypted image is identical to the original image.")
    else:
        print("Error: The decrypted image differs from the original image.")

    # Convert encrypted_img1 to uint8 type
    encrypted_img1_uint8 = np.array(encrypted_img1, dtype=np.uint8)

    # Display images
    cv2.imshow("Original Image", img)
    cv2.imshow("Encrypted Image", encrypted_img1_uint8)
    cv2.imshow("Decrypted Image", decrypted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
