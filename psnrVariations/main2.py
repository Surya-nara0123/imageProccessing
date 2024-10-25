import random
import numpy as np
import cv2
import time
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

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

# Emperor Penguin Optimizer for key generation
def epo_optimize(img, population_size, iterations, block_size):
    height, width = img.shape
    best_key = None
    best_psnr = float("inf")

    # Initialize penguin population (keys)
    population = [get_random_bytes(block_size) for _ in range(population_size)]

    avg_psnr_values = []
    for _ in range(iterations):
        # Track the best psnr and corresponding key in the population
        psnr_values = []
        for key in population:
            encrypted_img1 = np.zeros((height, width), dtype=object)
            for i in range(height):
                for j in range(width):
                    msg = chr(img[i][j]).encode()

                    en_msg, iv = aes_encrypt(msg, key, block_size)
                    encrypted_byte = en_msg[0]  # First byte of encrypted message

                    encrypted_img1[i][j] = (
                        encrypted_byte % 256
                    )  # Store the first encrypted byte

            # Calculate psnr for the current key
            psnr = calculate_psnr(img, encrypted_img1)

            # Update the best key if this one has a higher psnr
            if psnr < best_psnr:
                best_psnr = psnr
                best_key = key

            print("#" * 50)
            print("Best initial value: ", best_key)
            print("Best psnr: ", best_psnr)
            print("#" * 50)
            psnr_values.append(best_psnr)
        psnr_values = set(psnr_values)
        avg_psnr_values.append(sum(psnr_values) / len(psnr_values))

        # Move penguins (keys) based on the best-performing solution
        for i in range(population_size):
            if random.random() < 0.3:  # Introduce random movement for diversity
                population[i] = get_random_bytes(block_size)
            else:
                # Move towards the best key by adding a small random tweak and ensure valid byte range
                tweak = bytearray(
                    (best_key[j] + random.randint(-5, 5)) % 256
                    for j in range(block_size)
                )
                population[i] = tweak

    # save the values in a file
    with open(f"psnr_values_aes_{population_size}_{iterations}.txt", "w") as f:
        for item in avg_psnr_values:
            f.write("%s\n" % item)
    return best_key, best_psnr


# AES Encryption function with IV return
def aes_encrypt(msg, key, block_size):
    iv = get_random_bytes(16)  # Random IV for encryption
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_msg = pad(msg, block_size)
    encrypted_msg = cipher.encrypt(padded_msg)
    return encrypted_msg, iv  # Return both encrypted message and IV


# AES Decryption function with IV parameter
def aes_decrypt(encrypted_msg, key, iv, block_size):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_msg = unpad(cipher.decrypt(encrypted_msg), block_size)
    return decrypted_msg


def main():
    img = cv2.imread("../../img.png", 0)
    height, width = img.shape
    encrypted_img1 = np.zeros((height, width), dtype=object)
    decrypted_img = np.zeros((height, width), dtype=np.uint8)
    block_size = 16  # AES block size (128 bits)

    currTime = time.time()

    # Emperor Penguin Optimization
    for i in range(2, 3):
        for j in range(3):
            best_key, best_psnr = epo_optimize(
                img, population_size=10*(i+1), iterations=50*(j+1), block_size=block_size
            )
            print(f"Best key: {best_key.hex()}, Best psnr: {best_psnr}")
            print("Time taken: ", time.time() - currTime)


if __name__ == "__main__":
    main()
