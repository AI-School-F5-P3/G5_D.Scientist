from passlib.context import CryptContext

# Crear el contexto para bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hashear la contraseña
hashed_password = pwd_context.hash("password")
print(hashed_password)
