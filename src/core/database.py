import pymssql
import time
from core.utils import log, secs
import sys

class Database:
    def __init__(self, config):
        self.config = config
        self.conn = None

    def connect(self):
        if self.conn is None:
            t0 = time.perf_counter()
            try:
                self.conn = pymssql.connect(
                    server=self.config["host"],
                    user=self.config["user"],
                    password=self.config["password"],
                    database=self.config["database"],
                    port=self.config["port"]
                )
                log("ok", "db.connect", time=secs(time.perf_counter() - t0))
                return True
            except Exception as e:
                log("error", "db.connect", level="ERROR", error=e)
                return False
        return True

    def query(self, sql, params=None, name=None, **log_kv):
        t0 = time.perf_counter()
        try:
            self.connect()
            if self.conn:
                cursor = self.conn.cursor(as_dict=True)
                if params:
                    cursor.execute(sql, params)
                else:
                    sql = sql.replace("%%", "%")
                    cursor.execute(sql)
                rows = cursor.fetchall()
                elapsed = time.perf_counter() - t0

                log("ok", "db.query", q=name or sys._getframe(1).f_code.co_name, **log_kv, rows=len(rows), time=secs(elapsed))
                return 1, rows, None
            elapsed = time.perf_counter() - t0
            log("error", "db.query", level="ERROR", reason="DB connect fail")
            return 0, None, "DB connect fail"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log("error", "db.query", level="ERROR", error=e, sql=sql[:80])
            return 0, None, str(e)

    def close(self):
        if self.conn:
            t0 = time.perf_counter()
            self.conn.close()
            self.conn = None
            log("ok", "db.close", time=secs(time.perf_counter() - t0))