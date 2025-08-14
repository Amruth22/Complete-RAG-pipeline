#!/usr/bin/env python3
"""
Main entry point for the RAG Pipeline
"""

import argparse
import sys
from src.rag_pipeline import RAGPipeline
from data.sample_story import get_sample_story

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Complete RAG Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--query", type=str, help="Ask a single question")
    parser.add_argument("--ingest", action="store_true", help="Ingest sample document")
    parser.add_argument("--test", action="store_true", help="Test all components")
    parser.add_argument("--stats", action="store_true", help="Show pipeline statistics")
    parser.add_argument("--reset", action="store_true", help="Reset the pipeline")
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    if args.test:
        print("Testing all pipeline components...")
        test_results = rag.test_components()
        
        print("\nTest Results:")
        for component, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {component}: {status}")
        
        if all(test_results.values()):
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed. Please check your configuration.")
        return
    
    if args.reset:
        rag.reset_pipeline()
        print("Pipeline reset completed!")
        return
    
    if args.ingest:
        print("Ingesting sample document...")
        sample_story = get_sample_story()
        stats = rag.ingest_documents([sample_story])
        
        print("\nIngestion completed!")
        print(f"Documents: {stats['total_documents']}")
        print(f"Chunks: {stats['total_chunks']}")
        print(f"Embeddings: {stats['total_embeddings']}")
        return
    
    if args.stats:
        stats = rag.get_pipeline_stats()
        print("Pipeline Statistics:")
        print(f"  Indexed: {stats['is_indexed']}")
        print(f"  Total embeddings: {stats['vector_store_stats']['total_embeddings']}")
        print(f"  Vector dimension: {stats['vector_store_stats']['dimension']}")
        print(f"  Configuration:")
        print(f"    Chunk size: {stats['config']['chunk_size']}")
        print(f"    Chunk overlap: {stats['config']['chunk_overlap']}")
        print(f"    Top-K results: {stats['config']['top_k_results']}")
        print(f"    LLM model: {stats['config']['llm_model']}")
        print(f"    Embedding model: {stats['config']['embedding_model']}")
        return
    
    if args.query:
        # Try to load existing index
        if not rag.load_existing_index():
            print("No existing index found. Please run with --ingest first.")
            return
        
        print(f"Query: {args.query}")
        result = rag.query(args.query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print("\nResponse:")
        print(result["response"])
        
        print(f"\nUsed {result['num_context_chunks']} context chunks")
        if result["similarity_scores"]:
            print(f"Top similarity scores: {[f'{score:.3f}' for score in result['similarity_scores'][:3]]}")
        return
    
    if args.demo:
        # Run the demo
        from examples.demo import main as demo_main
        demo_main()
        return
    
    # Interactive mode (default)
    print("RAG Pipeline - Interactive Mode")
    print("Commands:")
    print("  help    - Show this help")
    print("  ingest  - Ingest sample document")
    print("  stats   - Show pipeline statistics")
    print("  test    - Test all components")
    print("  reset   - Reset pipeline")
    print("  quit    - Exit")
    print("Or just type your question!")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("Available commands:")
                print("  ingest  - Ingest sample document")
                print("  stats   - Show pipeline statistics")
                print("  test    - Test all components")
                print("  reset   - Reset pipeline")
                print("  quit    - Exit")
                print("Or ask any question about the ingested documents!")
                continue
            
            if user_input.lower() == 'ingest':
                print("Ingesting sample document...")
                sample_story = get_sample_story()
                stats = rag.ingest_documents([sample_story])
                print(f"✅ Ingested {stats['total_chunks']} chunks from {stats['total_documents']} document(s)")
                continue
            
            if user_input.lower() == 'stats':
                stats = rag.get_pipeline_stats()
                print(f"Indexed: {stats['is_indexed']}")
                print(f"Total embeddings: {stats['vector_store_stats']['total_embeddings']}")
                continue
            
            if user_input.lower() == 'test':
                test_results = rag.test_components()
                for component, passed in test_results.items():
                    status = "✅" if passed else "❌"
                    print(f"  {status} {component}")
                continue
            
            if user_input.lower() == 'reset':
                rag.reset_pipeline()
                print("✅ Pipeline reset!")
                continue
            
            if not user_input:
                continue
            
            # Try to load index if not already loaded
            if not rag.is_indexed:
                if not rag.load_existing_index():
                    print("❌ No documents indexed. Please run 'ingest' first.")
                    continue
            
            # Process the query
            print("Processing...")
            result = rag.query(user_input)
            
            if "error" in result:
                print(f"❌ {result['error']}")
                continue
            
            print("\nResponse:")
            print(result["response"])
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()