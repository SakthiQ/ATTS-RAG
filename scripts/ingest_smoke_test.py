import os
import sys

# Allow running as a script from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.loader import DocumentLoader
from app.rag.chunker import DocumentChunker
from app.rag.vectorstore import VectorStoreManager
from app.rag.ingestion_guard import IngestionGuard
from app.rag.ingestion import ingest_file

def run_test():
    # 1. Create a dummy policy file for testing
    dummy_content = """
    OFFICE POLICY 2026
    
    1. LEAVE POLICY: Employees are entitled to 20 days of paid annual leave. 
    Applications must be submitted 2 weeks in advance via the HR portal.
    
    2. DRESS CODE: Business casual is required from Monday to Thursday. 
    Fridays are 'Casual Fridays'.
    
    3. REMOTE WORK: Employees can work from home 2 days per week with 
    manager approval.
    """

    test_file = "data/test_policy.txt"
    os.makedirs("data", exist_ok=True)
    with open(test_file, "w") as f:
        f.write(dummy_content)

    print(f"[OK] Created test file: {test_file}")

    # 2. Ingest the file
    print("[WAIT] Processing document...")
    loader = DocumentLoader()
    chunker = DocumentChunker()
    vsm = VectorStoreManager()
    guard = IngestionGuard()

    # Load -> Chunk -> Poisoning scan -> Add to Store
    summary = ingest_file(test_file, vsm, guard, loader, chunker)

    print(f"[OK] Ingestion result: {summary}")

    # 3. Test Search (The Magic Part)
    query = "How many days of leave do I get?"
    print(f"\n[QUERY] Searching for: '{query}'")

    results = vsm.search(query, k=1)

    if results:
        print("\n--- Search Result ---")
        print(f"Content: {results[0]['content'].strip()}")
        print(f"Source: {results[0]['metadata']['source']}")
        print("---------------------")
        print("\n[SUCCESS] It works! The system found the relevant policy using math/vectors.")
    else:
        print("[FAIL] Search failed to find anything.")

if __name__ == "__main__":
    run_test()
