from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

class InputData(BaseModel):
    inputs: List[List[float]]

@app.post("/invocations")
async def predict(data: InputData):
    result = [sum(row) for row in data.inputs]
    return {"predictions": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)