import psycopg
from psycopg.rows import dict_row
from datetime import date
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
API_KEY = os.getenv("API_KEY")

class Indicator(BaseModel):
    id: Optional[int] = None
    eur_ron: float
    usd_ron: float
    ircc: float
    robor: float
    dummy: int
    data: date

def db_conn():
    conn = psycopg.connect(
        user=DB_USER,
        dbname=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        row_factory=dict_row
    )
    c = conn.cursor()
    return conn, c

def auth(x_api_key):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid API KEY')

app = FastAPI()

@app.get('/', description='Buna ziua, va prezint lucrarea mea de licenta: prognoza cursului de schimb prin automatizarea analizei statistice a seriilor de timp')
async def root():
    return {"Lucrare": "Licenta"}

@app.get('/get_all')
async def get_all():
    conn, c = db_conn()
    c.execute("SELECT * FROM indicatori")
    res = c.fetchall()
    conn.close()
    return res

@app.get('/get_byId')
async def get_by_id(id: int):
    conn, c = db_conn()
    c.execute("SELECT * FROM indicatori WHERE id=%(id)s", {'id': id})
    res = c.fetchone()
    conn.close()
    return res

@app.post('/post', description="""data trebuie sa aiba acest format: YYYY-MM-DD (ex: 2026-01-01)""")
async def post(eur_ron: float, usd_ron: float, ircc: float, robor: float, dummy: int, data: date, x_api_key: str = Header(...)):
    auth(x_api_key)
    conn, c = db_conn()
    c.execute("INSERT INTO indicatori (eur_ron, usd_ron, ircc, robor, dummy, data) VALUES (%(eur_ron)s, %(usd_ron)s, %(ircc)s, %(robor)s, %(dummy)s, %(data)s)", {'eur_ron': eur_ron, 'usd_ron': usd_ron, 'ircc': ircc, 'robor': robor, 'dummy': dummy, 'data': data})
    conn.commit()
    conn.close()
    return {"operatie": "realizata"}

@app.post('/post_inBulk')
async def post_in_bulk(items: List[Indicator], x_api_key: str = Header()):
    auth(x_api_key)
    conn, c = db_conn()
    c.executemany("INSERT INTO indicatori (eur_ron, usd_ron, ircc, robor, dummy, data) VALUES (%(eur_ron)s, %(usd_ron)s, %(ircc)s, %(robor)s, %(dummy)s, %(data)s)", [item.model_dump(exclude={"id"}) for item in items])
    conn.commit()
    conn.close()
    return {"operatie": "realizata"}

@app.delete('/del_row_byId')
async def delete_row_by_id(id: int, x_api_key: str = Header(...)):
    auth(x_api_key)
    conn, c = db_conn()
    c.execute("DELETE FROM indicatori WHERE id=%(id)s", {'id': id})
    conn.commit()
    conn.close()
    return {"operatie": "realizata"}

@app.delete('/del_all_rows')
async def delete_all_rows(x_api_key: str = Header(...)):
    auth(x_api_key)
    conn, c = db_conn()
    c.execute("DELETE FROM indicatori")
    conn.commit()
    conn.close()
    return {"operatie": "realizata"}

