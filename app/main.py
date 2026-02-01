from fastapi import FastAPI
from .api import infer

app = FastAPI()

app.include_router(
    infer.router,
    prefix="/v1"
)

@app.get("/")
def home():
    return {"message": "Go to /docs for more"}


