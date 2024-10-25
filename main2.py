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
def epo_optimize(img, population_size, iterations, block_size):
    height, width = img.shape
    best_key = None
    best_uaci = -float("inf")

    # Initialize penguin population (keys)
    population = [get_random_bytes(block_size) for _ in range(population_size)]

    avg_uaci_values = []
    for _ in range(iterations):
        # Track the best UACI and corresponding key in the population
        uaci_values = []
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

            # Calculate UACI for the current key
            uaci = calculate_uaci(encrypted_img1, img)

            # Update the best key if this one has a higher UACI
            if uaci > best_uaci:
                best_uaci = uaci
                best_key = key

            print("#" * 50)
            print("Best initial value: ", best_key)
            print("Best UACI: ", best_uaci)
            print("#" * 50)
            uaci_values.append(best_uaci)
        uaci_values = set(uaci_values)
        avg_uaci_values.append(sum(uaci_values) / len(uaci_values))

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
    with open("uaci_values_aes_30.txt", "w") as f:
        for item in avg_uaci_values:
            f.write("%s\n" % item)
    return best_key, best_uaci


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
    img = cv2.imread("../img.png", 0)
    height, width = img.shape
    encrypted_img1 = np.zeros((height, width), dtype=object)
    decrypted_img = np.zeros((height, width), dtype=np.uint8)
    block_size = 16  # AES block size (128 bits)

    currTime = time.time()

    # Emperor Penguin Optimization
    best_key, best_uaci = epo_optimize(
        img, population_size=10, iterations=200, block_size=block_size
    )
    print(f"Best key: {best_key.hex()}, Best UACI: {best_uaci}")

    # Encrypt and Decrypt the image using the best key
    for i in range(height):
        for j in range(width):
            msg = chr(img[i][j]).encode()

            en_msg, iv = aes_encrypt(
                msg, best_key, block_size
            )  # Get encrypted message and IV
            encrypted_img1[i][j] = en_msg[0] % 256  # Store the first encrypted byte

            dr_msg = aes_decrypt(
                en_msg, best_key, iv, block_size
            )  # Pass the IV for decryption
            decrypted_pixel_value = dr_msg[0] if len(dr_msg) > 0 else 0
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
