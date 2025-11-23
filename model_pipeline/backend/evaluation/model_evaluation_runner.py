import os
import sys
import asyncio
import uuid
import pandas as pd
import yaml
from pathlib import Path

# --- PATH SETUP ---
#Get the directory of this script: .../backend/evaluation
current_dir = os.path.dirname(os.path.abspath(__file__))
#Get the 'backend' directory: .../backend
backend_dir = os.path.abspath(os.path.join(current_dir, ".."))

#FORCE 'backend' to the top of sys.path
#This ensures 'agents' and 'logger_config' are found as top-level modules
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

print(f"✅ Added to sys.path: {backend_dir}")

# imports
try:
    #import 'agents' directly because we are "inside" the backend folder context
    from agents.graph import agent_graph
    from logger_config import get_logger # Verify this works too
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    print("Debug: Files in backend_dir:")
    print(os.listdir(backend_dir))
    sys.exit(1)

async def run_evaluation():
    # Load config
    config_path = os.path.join(current_dir, "config_eval.yaml")
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    cfg = yaml.safe_load(open(config_path))
    
    testset_path = cfg["testset"]["path"] 
    results_path = cfg["evaluation"]["results_path"] 
    
    # Handle relative paths in config (relative to repo root)
    repo_root = os.path.abspath(os.path.join(backend_dir, "../.."))
    
    if not os.path.isabs(testset_path):
        testset_path = os.path.join(repo_root, testset_path)
    if not os.path.isabs(results_path):
        results_path = os.path.join(repo_root, results_path)

    print(f"Loading test set from: {testset_path}")
    try:
        df = pd.read_parquet(testset_path)
    except FileNotFoundError:
        print(f"❌ Test set not found. Please run generate_validation_queries.py first.")
        return

    results = []

    print(f"Starting evaluation of {len(df)} queries...")

    for i, row in df.iterrows():
        question = row["question"]
        thread_id = str(uuid.uuid4())
        
        # Build initial state
        init_state = {
            "messages": [{"role": "user", "content": question}],
            "thread_id": thread_id,
            "hotel_id": "111418", # Default hotel ID for testing
            "user_id": "eval_user"
        }

        answer = ""
        retrieved_context = ""
        
        run_config = {"configurable": {"thread_id": thread_id}}
        
        try:
            print(f"Processing Q{i}: {question[:50]}...")
            
            # Run agent
            response = await agent_graph.ainvoke(init_state, config=run_config)
            
            # Extract response
            if response and response.get("messages") and response["messages"][-1].get("role") == "assistant":
                answer = response["messages"][-1]["content"]
            
            # Extract context (saved by our previous fixes)
            if response and "retrieved_context" in response:
                retrieved_context = response["retrieved_context"]
            
        except Exception as e:
            print(f"❌ Graph invocation error for Q{i}: {e}")
            answer = f"Error: {e}"
            retrieved_context = ""

        results.append({
            "question": question,
            "answer": answer,
            "context": retrieved_context
        })
        
        await asyncio.sleep(0.5)

    # Save results
    os.makedirs(Path(results_path).parent, exist_ok=True)
    pd.DataFrame(results).to_parquet(results_path, index=False)
    print(f"\n✅ Saved evaluation results to {results_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())