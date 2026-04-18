import os
import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "highly-secure-jwt-secret-key")
ALGORITHM = "HS256"

def lambda_handler(event, context):
    try:
        # Bypass authorization for public routes
        raw_path = event.get('rawPath', '')
        if raw_path in ['/api/login', '/api/signup']:
            return {"isAuthorized": True}
            
        # HTTP APIs pass headers in lowercase
        headers = event.get('headers', {})
        
        # API Gateway HTTP API uses single string headers
        auth_header = headers.get('authorization', '')
        
        if not auth_header.startswith('Bearer '):
            print("Missing or invalid authorization header format")
            return {"isAuthorized": False}
            
        token = auth_header.split(' ')[1]
        
        # Validate JWT using the exact same secret and algo as backend/auth.py
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        print(f"Successfully authorized user: {payload.get('sub')}")
        
        return {
            "isAuthorized": True,
            "context": {
                "sub": payload.get("sub", ""),
                "role": "user"
            }
        }
        
    except jwt.ExpiredSignatureError:
        print("Token expired")
        return {"isAuthorized": False}
    except jwt.InvalidTokenError:
        print("Invalid token")
        return {"isAuthorized": False}
    except Exception as e:
        print(f"Error validating token: {str(e)}")
        return {"isAuthorized": False}
