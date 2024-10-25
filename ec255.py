import numpy as np
import cv2
import time
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import random

def gen_key():
    # Generate a Curve25519 private key
    private_key = x25519.X25519PrivateKey.generate()
    return private_key

def get_public_key(private_key):
    # Get the corresponding public key
    return private_key.public_key()

def encrypt(msg, private_key, public_key):
    en_msg = []
    # Generate a random shared secret
    shared_secret = private_key.exchange(public_key)
    
    # Use the shared secret to encrypt the message
    for char in msg:
        encrypted_val = (ord(char) + sum(shared_secret)) % 256  # Simple example of encryption
        en_msg.append(encrypted_val)
    return en_msg

def decrypt(en_msg, private_key, public_key):
    dr_msg = []
    # Generate a shared secret using the private key
    shared_secret = private_key.exchange(public_key)
    
    # Use the shared secret to decrypt the message
    for num in en_msg:
        decrypted_val = (num - sum(shared_secret)) % 256  # Simple example of decryption
        dr_msg.append(chr(decrypted_val))
    return dr_msg

def main():
    img = cv2.imread("../img.png", 0)
    height, width = img.shape
    encrypted_img = np.zeros((height, width), dtype=np.uint8)
    decrypted_img = np.zeros((height, width), dtype=np.uint8)

    currTime = time.time()

    # Generate private and public keys
    private_key = gen_key()
    public_key = get_public_key(private_key)

    for i in range(height):
        for j in range(width):
            msg = chr(img[i][j])
            # print(msg)
            en_msg = encrypt(msg, private_key, public_key)
            encrypted_img[i][j] = en_msg[0]
            # print(en_msg)

            dr_msg = decrypt(en_msg, private_key, public_key)
            # print(dr_msg)
            decrypted_pixel_value = ord(dr_msg[0])
            decrypted_img[i, j] = decrypted_pixel_value

    print("Time taken: ", time.time() - currTime)

    # Compare original and decrypted images
    if np.array_equal(img, decrypted_img):
        print("Success: The decrypted image is identical to the original image.")
    else:
        print("Error: The decrypted image differs from the original image.")

    # Calculate UACI for the encrypted image
    uaci = 0
    for i in range(height):
        for j in range(width):
            uaci += abs(encrypted_img[i, j]%256 - img[i, j])
            # print(encrypted_img[i, j][0], img[i, j])
    uaci /= height * width * 255 / 100
    print("UACI: ", uaci)

    # Convert encrypted_img to uint8 type for display

    # Display images
    cv2.imshow("Original Image", img)
    cv2.imshow("Encrypted Image", encrypted_img)
    cv2.imshow("Decrypted Image", decrypted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
