import os
from app.rag.loader import DocumentLoader
from loguru import logger

def test_multimodal_ingestion():
    loader = DocumentLoader()

    # Path to the existing PDF in the project
    pdf_path = "data/Beyond the Pilot - How Regulated Industries Can Successfully Scale AI.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"Test file not found: {pdf_path}")
        return

    logger.info(f"Starting multimodal ingestion test for: {pdf_path}")

    try:
        documents = loader.load_pdf(pdf_path)

        logger.info(f"Extracted {len(documents)} document segments.")

        # Real, ruled tables become their own documents (type == "table"); this sample PDF has
        # none (its "Pitfall / Impact" style boxes have no drawn grid lines -- see
        # docs/Knowledge_Ingestion.md), so this is expected to report 0.
        tables_found = 0
        for doc in documents:
            if doc["metadata"].get("type") == "table":
                tables_found += 1
                logger.info(f"--- Table Found on Page {doc['metadata']['page']} ---")
                print(doc["content"][-500:])

        logger.success(f"Test Complete. Total ruled tables found: {tables_found}")

    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_multimodal_ingestion()
