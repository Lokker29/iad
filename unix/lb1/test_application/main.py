from fastapi import FastAPI

app = FastAPI()

TEST_DB = {
    1: {"name": "Alexey", 'age': 22},
    2: {"name": "Olena", "age": 19}
}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/student/{item_id}")
def read_item(item_id: int):
    return TEST_DB.get(item_id, "Student does not found")
