# Workshop Data Pipeline
Quickstart guide for running the data pipeline for a Pinecone workshop. The end to end pipeline does the following:

1. Scrapes web content from a base URL.
1. Creates JSONL files that contain the text for each web scrape and key meta-data fields like "source" and "scrape_date"
1. Crease JSONL data files that contain the text chunks for each web scrape, vector representation for each chunk and key meta-data fields like "source" and "scrape_date"
1. Upserts the JSONL data files into Pinecone

#### IMPORTANT: Before running the data pipeline, you must create a serverless pinecone index.

### Step 0 - Install dependencies (not needed for Instruqt)

Setup virtual environment and install the required python packages. If you do not have poetry, you will need to install
with this command:

```
cd ./data
pip install poetry
poetry install
poetry shell
```

### Step 1 - Setup terminal
. myenv/bin/activate
poetry shell

### Step 2 - Set environment variables
Create a file named ```.env``` that has the following variables:

```
PINECONE_NAMESPACE=cnn
PINECONE_INDEX_NAME=workshop-[YOUR_NAME]
PINECONE_API_KEY=[YOUR_PINECONE_API_KEY]
AWS_TITAN_ENABLED=true
```

```AWS_TITAN_ENABLE``` will only use AWS Titan for embeddings if set to "true". Otherwise, it will default to E5-Large. 

**IMPORTANT: AWS Titan embeddings are 1536 dimensions and E5-Large embeddings are 1024 dimensions. You must set specify the correct dimension
value on index creation otherwise upsert will fail.**

### Step 3 - Run data pipeline - web scrape

```
python data_pipeline.py scrape
```

### Step 4 - View a web scrape JSONL file

```
cat ./jsonl/cnn_articles_us.jsonl
```

### Step 5 - Run data pipeline - pinecone upsert

```
python data_pipeline.py upsert
```

### Step 6 - Run data pipeline - print 3 test embeddings

```
python data_pipeline.py print
```

Copy one TEST EMBEDDING value so we can paste it into the Pinecone console for a manual query test
to validate TEST EMBEDDING METADATA and ANN query accuracy.

### Step 7 - Run a test query in Pinecone console

1. Login to the pinecone console
1. Select your workshop index
1. Click "Browser"
1. Enter your namespace in the "Namespace" field
1. Copy a vector from your JSONL data file (one of the "values" entries)
1. Paste the vector value in the "vector" field
1. Click "Query" button
1. Validate that the "text" metadata field in the first entry matches the "text" entry in your JSONL data file

### Step 8 - Run data pipeline - pinecone delete

This command will delete the vectors in your namespace. It will NOT delete the JSONL data that you web scraped. 

```
python data_pipeline.py delete
```

### MISC

#### VM Image Customizations

add the following to .bashrc
``` alias python='python3' ```

install the following packages + python dependencies
```
sudo apt install python3-pip
sudo apt install git
sudo apt install python3-venv
pip install poetry
git clone https://github.com/pinecone-field/pinecone-workshop.git 
cd /root/pinecone-workshop/data
python -m venv myenv
. myenv/bin/activate
poetry install

```
