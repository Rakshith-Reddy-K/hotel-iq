### **README: Evaluation Pipeline Setup & Usage**

This folder contains the automated evaluation framework for the HotelIQ chatbot. Follow these steps to set up the environment, download necessary data, and run the evaluation metrics.

#### **1. Environment Setup**

Before running any scripts, you must configure the backend environment.

1.  **Navigate to the backend directory:**

    ```bash
    cd model_pipeline/backend
    ```

2.  **Create the `.env` file:**
    Create a file named `.env` inside the `model_pipeline/backend` directory and add the following configuration:

    ```ini
    # OpenAI Configuration
    OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

    # Anthropic Configuration (Optional)
    ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>

    # Google Configuration
    GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>

    # Pinecone Configuration
    PINECONE_API_KEY=<YOUR_PINECONE_API_KEY>
    HOTEL_INDEX_NAME=hotel-iq-hotels
    REVIEWS_INDEX_NAME=hotel-iq-reviews

    # App Settings
    HOTEL_ID=111418
    CITY=boston
    LOG_LEVEL=info

    # GCP Configuration
    GCS_BUCKET_NAME=hotel_iq_bucket
    GCP_PROJECT_ID=hotel-iq
    GCP_REGION=us-east4
    GCP_INSTANCE_NAME=hotel-iq
    ```

3.  **Setup GCP Credentials:**

      * Create a `config` directory inside `backend`:
        ```bash
        mkdir -p config
        ```
      * Move your Google Cloud service account JSON file into this directory and rename it to `gcp-service-account.json` (or match the name expected by your local environment):
        ```bash
        mv /path/to/your/gcp-credentials.json config/gcp-service-account.json
        ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

#### **2. Data Initialization**

The evaluation scripts rely on local CSV files (`hotels.csv`, `reviews.csv`) to function correctly. You must run the main application once to download these files from Google Cloud Storage.

1.  **Run the backend server:**
    ```bash
    python main.py
    ```
2.  **Wait for initialization:** Watch the logs for "Downloading data files..." and "Startup complete\!".
3.  **Stop the server:** Once the data folder is populated, you can stop the server (Ctrl+C).

#### **3. Running the Evaluation**

Now that the environment is set and data is present, you can run the evaluation stages.

**Step A: Generate Test Questions**
Create a fresh set of validation queries based on the hotel data.

```bash
python evaluation/generate_validation_queries.py
```

  * **Output:** `evaluation/testsets/hotel_base_validation.parquet`

**Step B: Run the Chatbot (Generate Answers)**
Feed the questions into the agent graph to generate answers. This script runs the agent locally using the data downloaded in Step 2.

```bash
python evaluation/model_evaluation_runner.py
```

  * **Output:** `evaluation/results/hotel_base_validation_results.parquet`

**Step C: Grade the Results**
Run the Ragas evaluator to calculate metrics like Faithfulness, Answer Relevancy, and Context Recall.

```bash
python evaluation/ragas_eval_wrapper.py
```

  * **Output:** `evaluation/results/hotel_base_validation_ragas_metrics.parquet`
