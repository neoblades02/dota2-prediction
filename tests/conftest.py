import os
import tempfile
from pathlib import Path

# .env may point DATABASE_URL at the shared production Postgres; tests must never reach it.
os.environ["DOTA_DATABASE_URL"] = f"sqlite:///{(Path(tempfile.gettempdir()) / 'dota_predictor_test.db').as_posix()}"
