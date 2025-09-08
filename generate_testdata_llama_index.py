from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from litellm_embeddings import LiteLLMEmbeddings
from vector_functions import get_lite_llm_model
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
import environ

# Load environment variables
env = environ.Env()
environ.Env.read_env()

def create_ragas_testdata():
    print("=== RAGAS Test Data Generation with LlamaIndex ===")
    
    try:
        # Load documents using LlamaIndex
        print("1. Loading PDF documents...")
        docs = SimpleDirectoryReader("./pdf").load_data()
        print(f"   Loaded {len(docs)} documents")
        
        # Split documents using LlamaIndex node parser
        print("2. Splitting documents...")
        node_parser = SimpleNodeParser.from_defaults(chunk_size=800, chunk_overlap=50)
        nodes = node_parser.get_nodes_from_documents(docs)
        print(f"   Split into {len(nodes)} nodes")
        
        # Use small subset
        test_nodes = nodes[:5]
        print(f"   Using {len(test_nodes)} nodes for test generation")
        
        # Print sample content to verify
        print("   Sample content:")
        for i, node in enumerate(test_nodes):
            print(f"   Node {i+1}: {node.text[:100]}...")
        
        # Setup models
        print("3. Setting up models...")
        
        # Get LangChain LLM and wrap it properly for RAGAS
        llm = get_lite_llm_model()
        ragas_llm = LangchainLLMWrapper(llm)
        print(f"   RAGAS LLM configured: {type(ragas_llm)}")
        
        # Setup embeddings - wrap LiteLLM embeddings for RAGAS
        litellm_embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")
        ragas_embeddings = LangchainEmbeddingsWrapper(litellm_embeddings)
        print(f"   RAGAS Embeddings configured: {type(ragas_embeddings)}")
        
        # Create test generator
        print("4. Creating test generator...")
        generator = TestsetGenerator(
            llm=ragas_llm,
            embedding_model=ragas_embeddings
        )
        print("   Generator created successfully")
        
        # Generate test data
        print("5. Generating test dataset (this may take a few minutes)...")
        dataset = generator.generate_with_llama_index_docs(
            documents=test_nodes,
            testset_size=3,
            raise_exceptions=False,
            with_debugging_logs=True
        )
        
        print("   Test generation completed!")
        
        # Convert to DataFrame and save
        print("6. Saving results...")
        dataset_df = dataset.to_pandas()
        dataset_df.to_csv("TestData_From_Ragas_Llama.csv", encoding="utf-8", index=False)
        
        print(f"✅ Success! Saved {len(dataset_df)} test cases to TestData_From_Ragas_Llama.csv")
        print(f"   Columns: {list(dataset_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_ragas_testdata()
    
    if not success:
        print("\n💡 To fix RAGAS issues:")
        print("1. Verify your LiteLLM endpoint is accessible")
        print("2. Check if documents contain sufficient content")
        print("3. Try: pip install ragas==0.2.10 for older stable version")
        print("4. Use generate_simple_testdata.py as alternative")