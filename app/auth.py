import json
import os
from dataclasses import dataclass
from typing import Dict, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError

load_dotenv()

from passlib.context import CryptContext

pwd_context=CryptContext(schemes=['bcrypt'])

@dataclass(frozen=True)
class AuthUser:
    username:str
    password_hash:str
    role:str

def password_hash(password:str)->str:
    return pwd_context.hash(password)

def verify_password(password_hash:str, password:str)->bool:
    return pwd_context.verify(secret=password, hash=password_hash)

def load_users_from_env()-> Dict[str, AuthUser]:
    raw_users = os.getenv("AUTH_USERS_JSON")

    if not raw_users:
        return {}
    
    parsed_users = json.loads(raw_users)
    users={}
    for item in parsed_users:
        user = AuthUser(
        username=item['username'],
        password_hash=item['password_hash'],
        role=item['role'].lower()
        )
        users[user.username]=user

    return users

def authenticate_user(username:str, password:str):
    users = load_users_from_env()

    if not users:
        return None
    
    user = users.get(username)

    if user is None:
        return None
    is_password_valid = verify_password(user.password_hash, password)

    if not is_password_valid:
        return None

    return user


def create_access_token(username:str, role:str):
    secret_key = os.getenv("JWT_SECRET_KEY")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    if not secret_key:
        raise ValueError("JWT_SECRET_KEY is not configured")

    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=expire_minutes
    )

    payload = {
        "sub": username,
        "role": role,
        "exp": expires_at
    }

    encoded_jwt = jwt.encode(
        payload,
        secret_key,
        algorithm
    )
    
    return encoded_jwt


def decode_access_token(token:str):
    secret_key=os.getenv("JWT_SECRET_KEY")
    algorithm=os.getenv("JWT_ALGORITHM")

    if not secret_key:
        raise ValueError("JWT_SECRET_KEY is not configured")

    try:
        payload = jwt.decode(token, secret_key, algorithms=algorithm)
    except JWTError:
        return None

    username = payload.get('sub')
    role=payload.get('role')

    if not username or not role:
        return None

    return {
        "username": username,
        "role": role,
    }
    

if __name__ == "__main__":

    print(create_access_token("Tony", "engineering"))