from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def get_checkpointer(db_path: str = 'data/checkpoints.db') -> SqliteSaver:
    """Factory function to create a checkpointer instance.
    args : db_path : str : path to sqlite database file
    return : SqliteSaver : checkpointer instance

    Why raw sqlite3 connection instead of from_conn_string():
    In newer versions of langgraph-checkpoint-sqlite, from_conn_string()
    returns a _GeneratorContextManager (needs `with` block), not a
    BaseCheckpointSaver. Passing it directly to graph.compile() raises
    TypeError. Using sqlite3.connect() gives us the actual connection
    object that SqliteSaver expects.

    check_same_thread=False is required because Streamlit runs across
    multiple threads and SQLite connections are thread-locked by default.
    """
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info(f"Checkpointer initialized with database at {db_path}")
        return checkpointer
    except Exception as e:
        logger.error(f"Failed to initialize checkpointer: {e}")
        raise e
    
def get_conversation_state(
    thread_id: str,
    checkpointer:SqliteSaver
)->Optional[str]:
    """Retrive the latest conversation state for a given thread.
        returns latest checkpoint state or None .
        """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        #get latest checkpoint state for this thread
        #Why : langgraph automarically resumes from latest state
        state = checkpointer.get(config)
        return state
    except Exception as e:
        logger.error(f"Failed to retrieve conversation state for thread {thread_id}: {e}")
        return None
    
def clear_thread(thread_id:str,checkpointer:SqliteSaver)->bool:
    """Retreive the latest conversation state for a given thread"""
    try: 
        conn = checkpointer.conn
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        
        deleted_count = cursor.rowcount
        logger.info(f"Cleared checkpoints for thread {thread_id} (deleted {deleted_count} rows)")        
        return deleted_count > 0
    except Exception as e:
        logger.error(f"Failed to clear checkpoints for thread {thread_id}: {e}")
        return False
    
def list_all_threads(checkpointer:SqliteSaver)->list[dict]:
        """List all unique threads with metadata like checkpoint count and last updated timestamp."""
        try:
            conn = checkpointer.conn
            cursor = conn.cursor()
            
            # Get unique threads with metadata
            cursor.execute("""
                SELECT 
                    thread_id,
                    COUNT(*) as checkpoint_count,
                    MAX(created_at) as last_updated
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY last_updated DESC
            """)
            
            threads = []
            for row in cursor.fetchall():
                threads.append({
                    "thread_id": row[0],
                    "checkpoint_count": row[1],
                    "last_updated": row[2]
                })
            
            return threads
    
        except Exception as e:
            logger.error(f"Failed to list threads: {e}")
            return []
