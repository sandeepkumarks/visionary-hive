
# Scalable Retrieval-Augmented Generation (RAG) System

## Team Members
 - Sandeep Kumar KS (030065592)
 - Nivetha Ammundi Magesh (031537023)
 - Sai Krishna Kapa (032208109)


## Tech Stack
- **Python**: Core language for backend development.
- **FAISS**: For high-dimensional similarity search and distributed retrieval.
- **Elasticsearch**: Datastore for document sections and metadata, supporting efficient retrieval.
- **FastAPI**: API framework for handling user queries and responses.
- **Flower**: Federated learning framework for retrieval model optimization across distributed nodes.
- **SentenceTransformers**: For embedding generation with pre-trained models.
- **Torch**: Used for model training and inference in the generative component.
- **Docker**: Containerization of services for easy deployment.

## Installation

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- Docker
- Elasticsearch (running on localhost or a specified host)
- FAISS (for similarity search)

### Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd visionary-hive/server
   ```

2. **Set Up Python Environment**
   Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   - Create a `.env` file in the project root with the following variables:
     ```
     ELASTICSEARCH_HOST=<elasticsearch_host>
     FAISS_INDEX_PATH=<path_to_faiss_index>
     FLASK_APP=<main_application_file>
     ```
   
4. **Run Docker Containers**
   For services like Elasticsearch and Flower, start Docker containers:
   ```bash
   docker-compose up
   ```

5. **Start the API Server**
   Launch the API server using FastAPI:
   ```bash
   uvicorn run_client:app --host 0.0.0.0 --port 8001
   ```
## Frontend Setup (React)

### Steps to Start React Client Server

1. **Navigate to the Client Directory**
   ```bash
   cd visionary-hive/client
   ```

2. **Install Dependencies**
   Ensure that you have `npm` installed. Then, install the necessary packages:
   ```bash
   npm install
   ```

3. **Start the React Development Server**
   Launch the React server:
   ```bash
   npm start
   ```

4. **Access the Frontend**
   Once the server is running, open your browser and go to:
   ```
   http://localhost:3000


## Usage

1. **Querying the RAG System**
   - Send a POST request to the API endpoint `/answer` with the following JSON payload:
     ```json
     {
       "user_query": "Enter your question here",
       "isRAGEnabled": true
     }
     ```
   - Example:
     ```bash
     curl -X POST "http://localhost:8001/answer" -H "Content-Type: application/json" -d '{"user_query": "Does the company offer tuition reimbursement?", "isRAGEnabled": true}'
     ```

2. **Federated Learning Workflow**
   - Use Flower to handle federated updates. Ensure that each node in the FAISS distributed network is running locally or in a Docker container to participate in federated model training.

3. **View Results**
   - Responses to queries will be returned as JSON, containing fields for both the retrieved context and generated answer.

---