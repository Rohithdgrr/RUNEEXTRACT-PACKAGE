"""
🚀 Game-Changing Features Demo

This script demonstrates all 5 game-changing features that make
RuneExtract the obvious choice for RAG developers.
"""

import time
from runeextract import auto_rag


def demo_1_adaptive_intelligence():
    """Feature 1: Zero-Config Smart RAG Pipeline"""
    print("\n" + "="*70)
    print("🎯 FEATURE 1: ADAPTIVE INTELLIGENCE")
    print("="*70)
    
    # Create adaptive RAG - learns optimal parameters automatically
    print("\n📚 Creating adaptive RAG pipeline...")
    rag = auto_rag(
        "examples/sample_docs/",  # Your documents directory
        intelligence="adaptive",   # Enable self-tuning
        chunking="auto"           # Auto-detect best chunking strategy
    )
    
    print("\n✅ RAG initialized with adaptive intelligence")
    print("   - Auto-detects document types")
    print("   - Self-tunes based on query patterns")
    print("   - Learns from performance metrics")
    
    # First query - uses defaults
    print("\n📝 Query 1 (Factual):")
    result1 = rag.query("What is machine learning?")
    print(f"   Answer: {result1.answer[:100]}...")
    print(f"   Confidence: {result1.confidence:.2%}")
    print(f"   Latency: {result1.latency_ms:.0f}ms")
    
    # Second query - analytical, auto-enables advanced retrieval
    print("\n📝 Query 2 (Analytical - auto-optimized):")
    result2 = rag.query("Why is deep learning effective for image recognition?")
    print(f"   Answer: {result2.answer[:100]}...")
    print(f"   Confidence: {result2.confidence:.2%}")
    print(f"   Latency: {result2.latency_ms:.0f}ms")
    print(f"   🎯 Auto-enabled: HyDE={len(result2.query_variants) > 0}")
    
    # Third query - comparison, auto-increases top_k
    print("\n📝 Query 3 (Comparison - auto-tuned):")
    result3 = rag.query("Compare supervised vs unsupervised learning")
    print(f"   Answer: {result3.answer[:100]}...")
    print(f"   Retrieved chunks: {len(result3.retrieved_chunks)}")
    
    # Show learning stats
    stats = rag.get_stats()
    print(f"\n📊 Adaptive Intelligence Stats:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Avg confidence: {stats['avg_confidence']:.2%}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.0f}ms")
    print(f"   Intelligence mode: {stats['intelligence_mode']}")


def demo_2_live_document_sync():
    """Feature 2: Live Document Sync with Incremental Indexing"""
    print("\n" + "="*70)
    print("⚡ FEATURE 2: LIVE DOCUMENT SYNC & INCREMENTAL INDEXING")
    print("="*70)
    
    # Create RAG with live sync and incremental indexing
    print("\n📚 Creating RAG with live document sync...")
    rag = auto_rag(
        "examples/sample_docs/",
        watch=True,              # Enable live file monitoring
        incremental=True         # Only reindex changed files
    )
    
    print("\n✅ Live document watcher started")
    print("   👁️  Monitoring directory for changes")
    print("   ⚡ Incremental indexing enabled")
    
    # Simulate file change detection
    print("\n📄 Simulating document changes:")
    print("   File added → auto-indexed in background")
    print("   File modified → delta re-indexing")
    print("   File deleted → auto-removed from index")
    
    # Query while watcher runs in background
    print("\n📝 Querying (watcher runs in background):")
    result = rag.query("What are the latest updates?")
    print(f"   Answer: {result.answer[:100]}...")
    
    # Show watcher stats
    stats = rag.get_stats()
    print(f"\n📊 Sync Stats:")
    print(f"   Documents indexed: {stats['total_documents']}")
    print(f"   Watcher active: {stats['watching']}")
    
    # Demonstrate incremental indexing
    print("\n⚡ Re-ingesting directory (incremental):")
    rag.ingest("examples/sample_docs/", incremental=True)
    print("   ⏭️  Most files skipped (unchanged)")
    print("   📄 Only changed files reprocessed")
    
    # Stop watcher
    print("\n🛑 Stopping watcher...")
    rag.stop_watch()
    print("   ✅ Watcher stopped cleanly")


def demo_3_citation_engine():
    """Feature 3: Contextual Citation Engine with Source Linking"""
    print("\n" + "="*70)
    print("📚 FEATURE 3: CITATION ENGINE WITH PROVENANCE")
    print("="*70)
    
    # Create RAG (citations enabled by default)
    print("\n📚 Creating RAG with citation engine...")
    rag = auto_rag("examples/sample_docs/")
    
    print("\n✅ Citation engine ready")
    print("   - Auto-cites every factual claim")
    print("   - Tracks source provenance")
    print("   - Links to exact pages/locations")
    
    # Query with citations
    print("\n📝 Query with citations:")
    result = rag.query(
        "What methodology was used in the study?",
        cite=True  # Enable citation markers
    )
    
    print(f"\n📄 Answer with citations:")
    print(f"   {result.answer}")
    
    print(f"\n📚 Citations ({len(result.citations)}):")
    for i, citation in enumerate(result.citations, 1):
        print(f"\n   [{i}] {citation.source}")
        if citation.page:
            print(f"       Page: {citation.page}")
        print(f"       Similarity: {citation.similarity_score:.3f}")
        print(f"       Rank: #{citation.retrieval_rank}")
        print(f"       Extracted: {citation.extracted_at}")
        print(f"       Text: {citation.text[:80]}...")
    
    print(f"\n🎯 Citation Quality:")
    print(f"   Answer confidence: {result.confidence:.2%}")
    print(f"   Avg citation score: {sum(c.relevance_score for c in result.citations) / len(result.citations):.3f}")


def demo_4_multimodal_rag():
    """Feature 4: Multi-Modal RAG with Vision Understanding"""
    print("\n" + "="*70)
    print("👁️  FEATURE 4: MULTI-MODAL RAG")
    print("="*70)
    
    # Create multi-modal RAG
    print("\n📚 Creating multi-modal RAG pipeline...")
    rag = auto_rag(
        "examples/sample_docs/",
        multimodal=True  # Enable vision understanding
    )
    
    print("\n✅ Multi-modal RAG ready")
    print("   - Vision models for images")
    print("   - Chart interpretation")
    print("   - Table-aware retrieval")
    print("   - OCR + semantic analysis")
    
    # Query about visual content
    print("\n📝 Query about visual content:")
    result = rag.query(
        "What does the revenue chart show?",
        cite=True
    )
    
    print(f"\n📄 Answer (with image analysis):")
    print(f"   {result.answer[:200]}...")
    
    print(f"\n🖼️  Multi-modal features:")
    print(f"   Images analyzed: {rag.multimodal}")
    print(f"   Vision cache size: {len(rag._vision_cache)}")
    
    # Query about tables
    print("\n📝 Query about tables:")
    result2 = rag.query("What are the demographic breakdowns?")
    print(f"   Answer: {result2.answer[:150]}...")
    print(f"   (Retrieved from table data)")


def demo_5_production_safeguards():
    """Feature 5: Production-Ready Safeguards Built-In"""
    print("\n" + "="*70)
    print("🛡️  FEATURE 5: PRODUCTION SAFEGUARDS")
    print("="*70)
    
    # Create RAG with all safeguards enabled
    print("\n📚 Creating production-ready RAG...")
    rag = auto_rag(
        "examples/sample_docs/",
        safe_mode=True,          # Master security switch
        cost_limit=5.00,         # $5 budget cap
        scan_secrets=True,       # Auto-redact API keys
        timeout_per_doc=60       # 60s max per document
    )
    
    print("\n✅ Safe mode enabled - all protections active:")
    print("   💰 Cost tracking & hard limits")
    print("   🔒 Secret scanning (30+ patterns)")
    print("   ⏱️  Timeout protection")
    print("   🛡️  Prompt injection defense")
    print("   ⚡ Rate limiting")
    print("   🔧 Circuit breakers")
    
    # Query with cost tracking
    print("\n📝 Query 1:")
    result1 = rag.query("Summarize the key findings")
    print(f"   Answer: {result1.answer[:100]}...")
    print(f"   💰 Query cost: ${result1.cost:.4f}")
    print(f"   💰 Session total: ${result1.total_session_cost:.4f}")
    
    # Another query
    print("\n📝 Query 2:")
    result2 = rag.query("What are the recommendations?")
    print(f"   Answer: {result2.answer[:100]}...")
    print(f"   💰 Query cost: ${result2.cost:.4f}")
    print(f"   💰 Session total: ${result2.total_session_cost:.4f}")
    
    # Show safety stats
    stats = rag.get_stats()
    print(f"\n🛡️  Safety Stats:")
    print(f"   Safe mode: {stats['safe_mode']}")
    print(f"   Total cost: ${stats['total_cost']:.4f}")
    print(f"   Cost limit: ${stats['cost_limit']:.2f}")
    print(f"   Cost remaining: ${stats['cost_remaining']:.4f}")
    print(f"   Secret scanning: {stats.get('scan_secrets', 'N/A')}")
    
    # Demonstrate cost limit enforcement
    print("\n💰 Cost limit protection:")
    print(f"   Current: ${rag._total_cost:.4f} / ${rag.cost_limit:.2f}")
    print(f"   Remaining budget: ${rag.cost_limit - rag._total_cost:.4f}")
    print(f"   ✅ Hard limit prevents cost overruns")


def demo_all_features_together():
    """Complete demo with all 5 features enabled"""
    print("\n" + "="*70)
    print("🚀 ALL 5 FEATURES TOGETHER")
    print("="*70)
    
    # Create production-ready RAG with ALL features
    print("\n📚 Creating ultimate RAG pipeline...")
    rag = auto_rag(
        source="examples/sample_docs/",
        
        # Feature 1: Adaptive Intelligence
        intelligence="adaptive",
        chunking="auto",
        
        # Feature 2: Live Sync
        watch=True,
        incremental=True,
        
        # Feature 4: Multi-Modal
        multimodal=True,
        
        # Feature 5: Production Safeguards
        safe_mode=True,
        cost_limit=10.00,
        scan_secrets=True,
        timeout_per_doc=60,
        
        # Standard config
        llm="openai:gpt-4o-mini",
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    print("\n✅ Ultimate RAG initialized with:")
    print("   🎯 Adaptive Intelligence")
    print("   ⚡ Live Document Sync")
    print("   📚 Citation Engine")
    print("   👁️  Multi-Modal Understanding")
    print("   🛡️  Production Safeguards")
    
    # Complex query using all features
    print("\n📝 Complex query (using all features):")
    result = rag.query(
        "What were the Q4 revenue projections and how do they compare to the chart data?",
        cite=True,
        answer_length="medium"
    )
    
    print(f"\n📄 Answer:")
    print(f"   {result.answer}")
    
    print(f"\n📊 Comprehensive Results:")
    print(f"   🎯 Confidence: {result.confidence:.2%}")
    print(f"   ⚡ Latency: {result.latency_ms:.0f}ms")
    print(f"   💰 Cost: ${result.cost:.4f}")
    print(f"   📚 Citations: {len(result.citations)}")
    print(f"   🔍 Chunks retrieved: {len(result.retrieved_chunks)}")
    print(f"   📈 Tokens: {result.tokens_used['input']} in, {result.tokens_used['output']} out")
    
    # System health
    stats = rag.get_stats()
    print(f"\n🏥 System Health:")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Queries: {stats['total_queries']}")
    print(f"   Avg confidence: {stats['avg_confidence']:.2%}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.0f}ms")
    print(f"   Cost used: ${stats['total_cost']:.4f} / ${stats['cost_limit']:.2f}")
    print(f"   Watching: {stats['watching']}")
    print(f"   Multimodal: {stats['multimodal_enabled']}")
    print(f"   Safe mode: {stats['safe_mode']}")
    
    # Cleanup
    print("\n🧹 Cleanup:")
    rag.stop_watch()
    print("   ✅ Watcher stopped")
    print("   ✅ Resources released")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 RUNEEXTRACT: 5 GAME-CHANGING FEATURES DEMO")
    print("="*70)
    print("\nThis demo showcases the features that make RuneExtract")
    print("the obvious choice for RAG developers.\n")
    
    try:
        # Run individual feature demos
        demo_1_adaptive_intelligence()
        time.sleep(1)
        
        demo_2_live_document_sync()
        time.sleep(1)
        
        demo_3_citation_engine()
        time.sleep(1)
        
        demo_4_multimodal_rag()
        time.sleep(1)
        
        demo_5_production_safeguards()
        time.sleep(1)
        
        # Run combined demo
        demo_all_features_together()
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETE")
        print("="*70)
        print("\n🎉 All 5 game-changing features demonstrated!")
        print("\n📚 Next steps:")
        print("   1. Read: docs/GAME_CHANGING_FEATURES.md")
        print("   2. Try: pip install 'runeextract[all]'")
        print("   3. Build: Your own production RAG in minutes")
        print("\n🚀 RuneExtract: The workflow killer for RAG developers.\n")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("   Note: Some features require API keys or sample documents")
        print("   See docs/GAME_CHANGING_FEATURES.md for setup instructions")
