import yaml
import pandas as pd
from ragas import evaluate
from datasets import Dataset 
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from pathlib import Path
import sys, os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables.")

# Initialize LLM Judge
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
)

# Initialize Embeddings 
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # 3072 dimensions
    openai_api_key=OPENAI_API_KEY
)

# LOAD DATA & PATHS
# Get the directory of this script to find config relative to it
current_dir = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(current_dir, "config_eval.yaml")

if not os.path.exists(cfg_path):
    logger.error(f"❌ Config not found at {cfg_path}")
    sys.exit(1)

cfg = yaml.safe_load(open(cfg_path))

# Resolve path to results file
results_path_rel = cfg["evaluation"]["results_path"]
repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))
results_file_path = results_path_rel

if not os.path.isabs(results_file_path):
    results_file_path = os.path.join(repo_root, results_path_rel)

try:
    print(f"Loading evaluation results from: {results_file_path}")
    df = pd.read_parquet(results_file_path)
except FileNotFoundError:
    logger.error(f"❌ Error: Results parquet not found at {results_file_path}. Did Stage B (model_evaluation_runner) run?")
    sys.exit(1)

# PREPARE DATASET 
# Ragas expects: question, answer, contexts, ground_truth
df_ragas = df[['question', 'answer', 'context']].copy()
df_ragas['contexts'] = df_ragas['context'].apply(lambda x: [x] if isinstance(x, str) and x else [])

# Note: We are using the model's answer as ground_truth placeholder to allow the script to run.
# For strict ContextRecall/Precision, you typically need human-annotated ground truths.
df_ragas['ground_truth'] = df_ragas['answer'] 

dataset = Dataset.from_pandas(df_ragas)

# METRIC BINDING AND EVALUATION 
metrics_to_run = [
    AnswerRelevancy(llm=llm, embeddings=embeddings),
    Faithfulness(llm=llm),
    ContextRecall(llm=llm), 
    ContextPrecision(llm=llm),
]

print("Starting RAGAS evaluation with OpenAI LLM and Embeddings...")

result = evaluate(
    dataset=dataset, 
    metrics=metrics_to_run,
    raise_exceptions=False 
)

print("\n✅ RAGAS metrics calculated:")
print(result)

# --- 5. SAVE RESULTS ---
output_file_path = Path(results_file_path).parent / "hotel_base_validation_ragas_metrics.parquet"

try:
    df_result = result.to_pandas()
except Exception as e:
    logger.error(f"Failed to convert RAGAS result to DataFrame: {e}. Outputting raw result.")
    df_result = pd.DataFrame([result])
    
df_result.to_parquet(output_file_path, index=False)
print(f"\nSaved ragas metrics to: {output_file_path}")