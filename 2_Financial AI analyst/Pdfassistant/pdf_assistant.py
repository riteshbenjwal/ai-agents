import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

try:
    knowledge_base = PDFUrlKnowledgeBase(
        urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=PgVector(
            table_name="recipes",
            db_url=db_url,
            search_type=SearchType.hybrid,
        ),
    )
    print("knowledge_base initialized.", knowledge_base)
    knowledge_base.load()
except Exception as e:
    print("Error loading knowledge base:", e)
    exit(1)

# Set up storage for the assistant
storage = PgAssistantStorage(
    table_name="pdf_assistant",
    db_url=db_url
)

# Defining the main function for the PDF assistant
def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    # If not starting a new session, fetch the first existing run ID
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    # Creating the assistant with the provided or new run ID
    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    # Printing the appropriate message based on the session status
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    # Launching the assistant CLI application
    assistant.cli_app(markdown=True)


if __name__=="__main__":
    typer.run(pdf_assistant)
