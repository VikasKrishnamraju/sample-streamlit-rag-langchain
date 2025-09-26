from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from litellm_embeddings import LiteLLMEmbeddings
from vector_functions import get_lite_llm_model
import os
import environ
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import apply_transforms, default_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers import MultiHopSpecificQuerySynthesizer

# Load environment variables
env = environ.Env()
environ.Env.read_env()

def create_ragas_testdata_kg():
    print("=== RAGAS Test Data Generation (Langchain) ===")
    
    try:
        # Setup models
        print(" Setting up models...")
        
        # Get LangChain LLM and wrap it properly for RAGAS
        llm = get_lite_llm_model()
        ragas_llm = LangchainLLMWrapper(llm)
        print(f"   RAGAS LLM configured: {type(ragas_llm)}")

        # Setup embeddings - wrap LiteLLM embeddings for RAGAS
        litellm_embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")
        ragas_embeddings = LangchainEmbeddingsWrapper(litellm_embeddings)
        print(f"   RAGAS Embeddings configured: {type(ragas_embeddings)}")
        
        # Load documents
        print("1. Loading PDF documents...")
        path = "pdf/"
        loader = PyPDFDirectoryLoader(path)
        docs = loader.load()
        print(f"   Loaded {len(docs)} documents")

        kg = KnowledgeGraph()

        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
                )      
            )   

        kg
        kg.save("knowledge_graph.json")

        # headline_extractor = HeadlinesExtractor(llm=ragas_llm, max_num=20)
        # headline_splitter = HeadlineSplitter(max_tokens=1500)
        # keyphrase_extractor = KeyphrasesExtractor(llm=ragas_llm)

        # transforms = [
        #     headline_extractor,
        #     headline_splitter,
        #     keyphrase_extractor
        # ]

        transforms = default_transforms(
            documents=docs,
            llm=ragas_llm,
            embedding_model=ragas_embeddings
            )

        apply_transforms(kg, transforms=transforms)

        persona_simple_user = Persona(
            name="Technical User",
            role_description="A technical user interested in AI, RAG systems, and document analysis."
        )
        personas = [persona_simple_user]

        # query_distribution = [
        #     (
        #         SingleHopSpecificQuerySynthesizer(llm=ragas_llm, property_name="headlines"),
        #         0.5,
        #     ),
        #     (
        #         SingleHopSpecificQuerySynthesizer(
        #         llm=ragas_llm, property_name="keyphrases"
        #     ),
        #         0.5,
        #     ),
        # ]

        
        # Alternative: If you want to try multihop, ensure proper knowledge graph setup
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.6),
            (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.4)
        ]
        
        # Create test generator
        print("4. Creating test generator...")
        generator = TestsetGenerator(
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
            knowledge_graph=kg,
            persona_list=[personas]
        )
        print("   Generator created successfully")
        
        # Generate test data
        print("5. Generating test dataset (this may take a few minutes)...")
        dataset = generator.generate(
            #documents=test_docs,
            testset_size=5,  # Reduced size for testing
            query_distribution=query_distribution,
            raise_exceptions=False,
            with_debugging_logs=True
        )
        
        print("   Test generation completed!")
        
        # Convert to DataFrame and save
        print("6. Saving results...")
        dataset_df = dataset.to_pandas()
        dataset_df.to_csv("TestData_From_Ragas_Langchain.csv", encoding="utf-8", index=False)

        print(f"✅ Success! Saved {len(dataset_df)} test cases to TestData_From_Ragas_Langchain.csv")
        print(f"   Columns: {list(dataset_df.columns)}")

        return dataset_df

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_ragas_testdata():
    print("=== RAGAS Test Data Generation (Langchain) ===")
    
    try:
        # Load documents
        print("1. Loading PDF documents...")
        path = "pdf/"
        loader = PyPDFDirectoryLoader(path)
        docs = loader.load()
        print(f"   Loaded {len(docs)} documents")
        
        # Split documents
        print("2. Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(docs)
        print(f"   Split into {len(split_docs)} chunks")
        
        # Use small subset
        test_docs = split_docs[:4]
        print(f"   Using {len(test_docs)} chunks for test generation")
        
        # Print sample content to verify
        print("   Sample content:")
        for i, doc in enumerate(test_docs):
            print(f"   Doc {i+1}: {doc.page_content[:100]}...")
        
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
        dataset = generator.generate_with_langchain_docs(
            documents=test_docs,
            testset_size=6,
            raise_exceptions=False,
            with_debugging_logs=True
        )
        
        print("   Test generation completed!")
        
        # Convert to DataFrame and save
        print("6. Saving results...")
        dataset_df = dataset.to_pandas()
        dataset_df.to_csv("TestData_From_Ragas_Langchain.csv", encoding="utf-8", index=False)
        
        print(f"✅ Success! Saved {len(dataset_df)} test cases to TestData_From_Ragas_Langchain.csv")
        print(f"   Columns: {list(dataset_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_ragas_testdata_kg()
    
    if not success:
        print("\n💡 To fix RAGAS issues:")
        print("1. Verify your LiteLLM endpoint is accessible")
        print("2. Check if documents contain sufficient content")
        print("3. Try: pip install ragas==0.2.10 for older stable version")
        print("4. Use generate_simple_testdata.py as alternative")