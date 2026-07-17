# FastAPI Overview

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Key Features

1. **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
2. **Fast to code**: Increase the speed to develop features by about 200% to 300%.
3. **Fewer bugs**: Reduce about 40% of human (developer) induced errors.
4. **Intuitive**: Great editor support. Completion everywhere. Less time debugging.

## Installation

To install FastAPI, run the following command:

```bash
pip install fastapi
```

You will also need an ASGI server, for production such as Uvicorn or Hypercorn.

```bash
pip install "uvicorn[standard]"
```

## Example: First Steps

Here is a minimal FastAPI application. It defines a single endpoint that returns a JSON greeting.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

To run the application, save the code in `main.py` and run:

```bash
uvicorn main:app --reload
```

## Path Parameters

You can declare path "parameters" or "variables" with the same syntax used by Python format strings:

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of the path parameter `item_id` will be passed to your function as the argument `item_id`. Because we declared the type as `int`, FastAPI will automatically validate the request and return an error if a non-integer is provided.
