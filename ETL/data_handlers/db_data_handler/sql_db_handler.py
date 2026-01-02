import sqlite3
from typing import Any, List
from data_handlers.db_data_handler.db_abstract import AbstractDBHandler


class SQLDBHandler(AbstractDBHandler):

    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        self.conn = None

    def connect(self) -> None:
        self.conn = sqlite3.connect(self.conn_str)

    def query(self, sql: str) -> List[Any]:
        cursor = self.conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def execute(self, sql: str, params: tuple | None = None):
        cursor = self.conn.cursor()
        cursor.execute(sql, params or ())
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
