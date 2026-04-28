import sys
print('Loading os')
import os
print('Loading datetime')
from datetime import datetime, timedelta
print('Loading typing')
from typing import Optional
print('Loading fastapi')
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
print('Loading bcrypt')
import bcrypt
print('Loading jwt')
import jwt
print('Loading pydantic')
from pydantic import BaseModel
print('Loading sqlalchemy')
import sqlalchemy
print('Loading create_engine')
from sqlalchemy import create_engine, Column, Integer, String
print('Loading orm')
from sqlalchemy.orm import declarative_base, sessionmaker, Session
print('Done auth trace')
