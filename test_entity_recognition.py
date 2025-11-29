
import sys
import os
sys.path.append(os.getcwd())
from src.utils.query_understanding import QueryUnderstanding

qu = QueryUnderstanding()
queries = [
    "Does metformin cause headaches?",
    "Is nausea a side effect of aspirin?",
    "tell me if advil causes stomach pain"
]

for q in queries:
    print(f"Query: {q}")
    entities = qu.extract_entities(q)
    print(f"Entities: {entities}")
