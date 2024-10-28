from datetime import datetime, timedelta
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "secret_key_default")  # Clave de encriptación JWT
ALGORITHM = os.getenv("ALGORITHM", "HS256")  # Algoritmo JWT
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    """
    Crea un token de acceso JWT que expira en `ACCESS_TOKEN_EXPIRE_MINUTES` minutos.
    
    Args:
        data (dict): Datos a codificar en el token.
        
    Returns:
        str: Token JWT.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    """
    Verifica un token JWT y decodifica su contenido.
    
    Args:
        token (str): Token JWT a verificar.
        credentials_exception (HTTPException): Excepción en caso de error de verificación.
        
    Returns:
        str: Usuario decodificado del token, o excepción si falla.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception
