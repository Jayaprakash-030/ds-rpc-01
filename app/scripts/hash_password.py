from passlib.context import CryptContext
from app.auth import password_hash
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", type=str, help="Write the password to generate hash password for it")
    args = parser.parse_args()
    print(password_hash(args.password))

