# src/redis_vector_store.py
from __future__ import annotations
import json, time
from typing import Any, Iterable, List, Dict, Optional
import numpy as np
import redis
VECTOR_FIELD = "embedding"
TEXT_FIELD = "text"
ID_FIELD = "id"
META_FIELD = "meta"

class RedisVectorStore:
    def __init__(
        self, *, host: str, port: int, password: str | None, ssl: bool,
        index_name: str, key_prefix: str, dims: int,
        distance_metric: str = "COSINE", dtype: str = "FLOAT32",
        hnsw_m: int = 16, hnsw_ef_construction: int = 200, hnsw_ef_runtime: int = 10,
    ) -> None:
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.dims = int(dims)
        self.distance_metric = distance_metric
        self.dtype = dtype
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_runtime = hnsw_ef_runtime
        self.r = redis.Redis(
            host=host,
            port=port,
            password=password,
            ssl=ssl,
            decode_responses=False,
            socket_connect_timeout=5,  # max 5s per connettersi
            socket_timeout=15,  # max 15s per comando
            retry_on_timeout=True,
            health_check_interval=15,
            socket_keepalive=True,
        )

    def _ping_with_retries(self, attempts=5, delay=1.0):
        last = None
        for i in range(attempts):
            try:
                self.r.ping()
                return True
            except Exception as e:
                last = e
                time.sleep(delay)
                delay *= 2  # backoff esponenziale
        raise RuntimeError(f"Connessione Redis fallita: {last}")

    def create_index(self, recreate: bool = False) -> None:
        t0 = time.perf_counter()
        # 1) PING (diagnostica)
        try:
            self.r.ping()
        except Exception as e:
            raise RuntimeError(f"Connessione Redis fallita: {e}")
        t_ping = time.perf_counter()

        # 2) DROP (se richiesto)
        if recreate:
            try:
                t_drop0 = time.perf_counter()
                # Se vuoi anche rimuovere i documenti indicizzati: aggiungi "DD"
                self.r.execute_command("FT.DROPINDEX", self.index_name)  # oppure: , "DD"
                t_drop1 = time.perf_counter()
                print(f"[IDX] DROP in {t_drop1 - t_drop0:.2f}s")
            except redis.ResponseError:
                pass

        # 3) CREATE (con opzioni)
        try:
            t_create0 = time.perf_counter()
            schema = [
                VECTOR_FIELD, "VECTOR", "HNSW", 6,
                "TYPE", self.dtype, "DIM", self.dims,
                "DISTANCE_METRIC", self.distance_metric,
                "EF_RUNTIME", self.hnsw_ef_runtime, "M", self.hnsw_m,
            ]
            # campi testuali
            schema += [TEXT_FIELD, "TEXT", "NOSTEM", META_FIELD, "TEXT", "NOSTEM", ID_FIELD, "TAG"]

            # ⚠️ Se hai già molte chiavi con quel PREFIX, NOINITIALSCAN evita la scansione iniziale
            self.r.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", self.key_prefix,
                "SCHEMA",
                VECTOR_FIELD, "VECTOR", "HNSW", "6",
                "TYPE", self.dtype,
                "DIM", self.dims,
                "DISTANCE_METRIC", self.distance_metric,
                #"M", self.hnsw_m,
                #"EF_CONSTRUCTION", self.hnsw_ef_construction,
                TEXT_FIELD, "TEXT", "NOSTEM",
                META_FIELD, "TEXT", "NOSTEM",
                ID_FIELD, "TAG"
            )

            t_create1 = time.perf_counter()
            print(f"[IDX] CREATE in {t_create1 - t_create0:.2f}s")
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                print("[IDX] Index already exists (skip CREATE)")
            else:
                raise

        print(f"[IDX] PING {t_ping - t0:.2f}s • TOTAL {time.perf_counter() - t0:.2f}s")

    def upsert(self, ids: List[str], embeddings: List[Iterable[float]],
               texts: Optional[List[str]] = None, metadatas: Optional[List[Dict[str, Any]]] = None,
               pipeline_size: int = 100) -> int:
        assert len(ids) == len(embeddings)
        n = len(ids)
        texts = texts or [""] * n
        metadatas = metadatas or [{}] * n
        pipe = self.r.pipeline(transaction=False)
        written = 0
        for i, _id in enumerate(ids):
            key = f"{self.key_prefix}{_id}"
            vec = np.asarray(list(embeddings[i]), dtype=np.float32).tobytes()
            pipe.hset(key, mapping={
                ID_FIELD: _id, VECTOR_FIELD: vec, TEXT_FIELD: texts[i] or "",
                META_FIELD: json.dumps(metadatas[i] or {}), "created_at": str(int(time.time())),
            })
            written += 1
            if written % pipeline_size == 0:
                pipe.execute()
        if written % pipeline_size:
            pipe.execute()
        return written

    def query(self, embedding: Iterable[float], k: int = 5,
              filter_expr: Optional[str] = None,
              return_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        vec = np.asarray(list(embedding), dtype=np.float32).tobytes()
        base = "*" if not filter_expr else f"({filter_expr})"
        q = f"{base}=>[KNN {k} @{VECTOR_FIELD} $vec AS score]"
        params = ["vec", vec]
        fields = return_fields or [ID_FIELD, TEXT_FIELD, META_FIELD, "score"]
        res = self.r.execute_command(
            "FT.SEARCH", self.index_name, q,
            "PARAMS", len(params), *params,
            "RETURN", len(fields), *fields,
            "SORTBY", "score", "DIALECT", 2,
        )
        items: List[Dict[str, Any]] = []
        if isinstance(res, list) and len(res) > 1:
            it = iter(res[1:])
            for key in it:
                doc = next(it, None)
                if not isinstance(doc, list):  # safety
                    continue
                obj: Dict[str, Any] = {"key": key.decode() if isinstance(key, (bytes, bytearray)) else str(key)}
                for j in range(0, len(doc), 2):
                    fname = doc[j].decode() if isinstance(doc[j], (bytes, bytearray)) else str(doc[j])
                    fval = doc[j+1]
                    if fname == META_FIELD:
                        try:
                            fval = json.loads(fval.decode() if isinstance(fval, (bytes, bytearray)) else fval)
                        except Exception:
                            pass
                    elif fname in (ID_FIELD, TEXT_FIELD):
                        fval = fval.decode() if isinstance(fval, (bytes, bytearray)) else fval
                    elif fname == "score":
                        try:
                            fval = float(fval.decode() if isinstance(fval, (bytes, bytearray)) else fval)
                        except Exception:
                            pass
                    obj[fname] = fval
                items.append(obj)
        return items

    def delete(self, ids: List[str]) -> int:
        pipe = self.r.pipeline(transaction=False)
        for _id in ids:
            pipe.delete(f"{self.key_prefix}{_id}")
        res = pipe.execute()
        return sum(1 for r in res if r)