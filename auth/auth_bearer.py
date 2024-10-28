from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from .auth_handler import verify_token
from database.db_operations import get_user_by_email

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Obtiene el usuario actual a partir del token JWT. Decodifica el token
    y recupera el objeto completo del usuario de la base de datos.
    
    Args:
        token (str): Token JWT de autenticación.
    
    Returns:
        dict: Objeto completo del usuario, incluyendo todos sus detalles.
    
    Raises:
        HTTPException: Si el token no es válido o no se encuentra el usuario.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar sus credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decodificar el token para obtener el email del usuario
    email = verify_token(token, credentials_exception)
    
    # Obtener el objeto completo del usuario usando el email
    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception
    
    # Devolver el usuario completo en lugar del email solo
    return user
