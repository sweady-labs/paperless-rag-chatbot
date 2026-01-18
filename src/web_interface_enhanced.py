#!/usr/bin/env python3
"""
Enhanced Web interface with real-time monitoring dashboard.

Features:
- Chat interface
- Real-time KPI dashboard
- Query metrics visualization
- System health monitoring
"""
import gradio as gr
import requests
import time
import json
from datetime import datetime

API_URL = "http://localhost:8001"


def query_chatbot(message, history):
    """Query the RAG chatbot with monitoring."""
    try:
        # Use regular endpoint for Gradio (faster and more reliable)
        response = requests.post(
            f"{API_URL}/query",
            json={"question": message},
            timeout=120  # 120s timeout for long table generation and explanations
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['answer']
            
            # Add sources with clickable links
            if result.get('sources'):
                answer += "\n\n**📄 Sources:**\n"
                for i, source in enumerate(result['sources'][:3], 1):
                    score = source.get('score', 0) * 100
                    title = source.get('title', 'Unknown')
                    url = source.get('url', '')
                    vector_score = source.get('vector_score')
                    bm25_score = source.get('bm25_score')
                    
                    answer += f"{i}. **{title}** ({score:.0f}%)\n"
                    if vector_score is not None and bm25_score is not None:
                        answer += f"   Vector: {vector_score:.3f} | BM25: {bm25_score:.3f}\n"
                    if url:
                        answer += f"   {url}\n"
            
            # Add performance metrics
            if result.get('metrics'):
                metrics = result['metrics']
                answer += f"\n\n**⚡ Performance:**\n"
                answer += f"- Total: {metrics.get('total_duration_ms', 0):.0f}ms\n"
                answer += f"- Search: {metrics.get('search_duration_ms', 0):.0f}ms\n"
                answer += f"- LLM: {metrics.get('llm_duration_ms', 0):.0f}ms ({metrics.get('tokens_per_sec', 0):.1f} tok/s)\n"
                answer += f"- Sources: {metrics.get('num_sources', 0)} (top score: {metrics.get('top_score', 0):.3f})"
            
            return answer
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Check if Ollama and API server are running."
    except requests.exceptions.ConnectionError:
        return "🔌 Cannot connect to API server. Run the API server first."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_health_status():
    """Get server health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            collection = data.get('collection', {})
            config = data.get('config', {})
            
            status = "🟢 **Online**\n\n"
            status += f"**Vector DB:**\n"
            status += f"- Chunks: {collection.get('points_count', 0):,}\n"
            status += f"- Collection: {collection.get('name', 'unknown')}\n\n"
            status += f"**Models:**\n"
            status += f"- LLM: {config.get('llm_model', 'unknown')}\n"
            status += f"- Embeddings: {config.get('embedding_model', 'unknown')}\n\n"
            status += f"**Search:**\n"
            status += f"- Hybrid: {'✅' if config.get('use_hybrid', False) else '❌'}\n"
            status += f"- Weights: {config.get('dense_weight', 0.6):.1f} / {config.get('sparse_weight', 0.4):.1f}"
            
            return status
        else:
            return "🟡 Server responding but has issues"
    except requests.exceptions.ConnectionError:
        return "🔴 **Offline**\n\nServer not running.\nStart with: `uvicorn src.api.server:app`"
    except Exception as e:
        return f"🔴 Error: {str(e)}"


def get_live_metrics():
    """Get live metrics from API."""
    try:
        response = requests.get(f"{API_URL}/metrics/live", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            
            # Format metrics
            metrics_text = "**📊 Session Statistics**\n\n"
            
            total = stats.get('total_queries', 0)
            metrics_text += f"**Queries:** {total}\n\n"
            
            if total > 0:
                latency = stats.get('latency', {})
                metrics_text += f"**Latency:**\n"
                metrics_text += f"- Avg: {latency.get('mean', 0):.0f}ms\n"
                metrics_text += f"- P50: {latency.get('p50', 0):.0f}ms\n"
                metrics_text += f"- P95: {latency.get('p95', 0):.0f}ms\n"
                metrics_text += f"- P99: {latency.get('p99', 0):.0f}ms\n\n"
                
                quality = stats.get('quality', {})
                metrics_text += f"**Quality:**\n"
                metrics_text += f"- Success: {quality.get('success_rate', 0):.0f}%\n"
                metrics_text += f"- Avg Score: {quality.get('avg_top_score', 0):.3f}\n"
                metrics_text += f"- Avg Sources: {quality.get('avg_num_sources', 0):.1f}\n\n"
                
                tokens = stats.get('tokens', {})
                metrics_text += f"**LLM:**\n"
                metrics_text += f"- Speed: {tokens.get('avg_per_sec', 0):.1f} tok/s\n"
                metrics_text += f"- Input: {tokens.get('total_input', 0):,}\n"
                metrics_text += f"- Output: {tokens.get('total_output', 0):,}\n\n"
                
                cache = stats.get('cache', {})
                metrics_text += f"**Cache:**\n"
                metrics_text += f"- Hit Rate: {cache.get('hit_rate', 0):.1f}%\n"
                metrics_text += f"- Hits: {cache.get('hits', 0)}\n"
                metrics_text += f"- Misses: {cache.get('misses', 0)}"
            
            return metrics_text
        else:
            return "⚠️ Could not fetch metrics"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_recent_queries_display():
    """Get recent queries for display."""
    try:
        response = requests.get(f"{API_URL}/metrics/recent?limit=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            queries = data.get('queries', [])
            
            if not queries:
                return "No queries yet"
            
            display = "**📜 Recent Queries**\n\n"
            for i, q in enumerate(queries, 1):
                query_text = q.get('query_text', 'N/A')
                duration = q.get('total_duration_ms', 0)
                score = q.get('top_score', 0)
                timestamp = q.get('timestamp', '')
                
                # Truncate long queries
                if len(query_text) > 40:
                    query_text = query_text[:37] + "..."
                
                display += f"{i}. \"{query_text}\"\n"
                display += f"   ⏱️ {duration:.0f}ms | 📊 {score:.3f}\n"
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        display += f"   🕐 {dt.strftime('%H:%M:%S')}\n"
                    except:
                        pass
                display += "\n"
            
            return display
        else:
            return "⚠️ Could not fetch recent queries"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_24h_statistics():
    """Get 24-hour statistics."""
    try:
        response = requests.get(f"{API_URL}/metrics/statistics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('last_24h', {})
            
            total = stats.get('total_queries', 0)
            if total == 0:
                return "No queries in last 24 hours"
            
            display = "**📈 Last 24 Hours**\n\n"
            display += f"**Total Queries:** {total}\n\n"
            
            latency = stats.get('latency', {})
            display += f"**Latency:**\n"
            display += f"- Avg: {latency.get('avg_ms', 0):.0f}ms\n"
            display += f"- Min: {latency.get('min_ms', 0):.0f}ms\n"
            display += f"- Max: {latency.get('max_ms', 0):.0f}ms\n\n"
            
            quality = stats.get('quality', {})
            display += f"**Quality:**\n"
            display += f"- Success: {quality.get('success_rate', 0):.0f}%\n"
            display += f"- Avg Score: {quality.get('avg_top_score', 0):.3f}\n\n"
            
            cache = stats.get('cache', {})
            display += f"**Cache:**\n"
            display += f"- Hit Rate: {cache.get('hit_rate', 0):.1f}%"
            
            return display
        else:
            return "⚠️ Could not fetch 24h statistics"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create enhanced Gradio interface
with gr.Blocks(title="Paperless RAG Chatbot - Monitoring Dashboard") as demo:
    gr.Markdown("# 📚 Paperless RAG Chatbot")
    gr.Markdown("Real-time document Q&A with comprehensive monitoring")
    
    with gr.Row():
        # Left column: Chat interface (70%)
        with gr.Column(scale=7):
            gr.Markdown("### 💬 Chat Interface")
            chatbot = gr.ChatInterface(
                query_chatbot,
                examples=[
                    "Wann ist Mavi geboren?",
                    "Lohnsteuerbescheinigung 2024",
                    "Welche Behandlungen hatte Luke?",
                    "Zeige mir Stromrechnungen von 2024",
                    "Gibt es Dokumente vom Standesamt?"
                ]
            )
        
        # Right column: Monitoring Dashboard (30%)
        with gr.Column(scale=3):
            gr.Markdown("### 🎛️ Monitoring Dashboard")
            
            # Server Status
            with gr.Group():
                gr.Markdown("#### 🔌 Server Status")
                health_display = gr.Markdown(value="Loading...")
                health_refresh = gr.Button("🔄 Refresh", size="sm")
            
            gr.Markdown("---")
            
            # Live Metrics
            with gr.Group():
                gr.Markdown("#### 📊 Live Metrics")
                metrics_display = gr.Markdown(value="Loading...")
                metrics_refresh = gr.Button("🔄 Refresh", size="sm")
            
            gr.Markdown("---")
            
            # Recent Queries
            with gr.Group():
                gr.Markdown("#### 📜 Recent Queries")
                recent_display = gr.Markdown(value="Loading...")
                recent_refresh = gr.Button("🔄 Refresh", size="sm")
            
            gr.Markdown("---")
            
            # 24h Statistics
            with gr.Group():
                gr.Markdown("#### 📈 24H Statistics")
                stats_24h_display = gr.Markdown(value="Loading...")
                stats_24h_refresh = gr.Button("🔄 Refresh", size="sm")
            
            gr.Markdown("---")
            
            # Quick Links
            gr.Markdown("#### 🔗 Quick Links")
            gr.Markdown("""
            - [API Docs](http://localhost:8001/docs)
            - [Metrics API](http://localhost:8001/metrics/live)
            - [Health Check](http://localhost:8001/health)
            """)
    
    # Wire up refresh buttons
    health_refresh.click(get_health_status, outputs=health_display)
    metrics_refresh.click(get_live_metrics, outputs=metrics_display)
    recent_refresh.click(get_recent_queries_display, outputs=recent_display)
    stats_24h_refresh.click(get_24h_statistics, outputs=stats_24h_display)
    
    # Auto-load on startup
    demo.load(get_health_status, outputs=health_display)
    demo.load(get_live_metrics, outputs=metrics_display)
    demo.load(get_recent_queries_display, outputs=recent_display)
    demo.load(get_24h_statistics, outputs=stats_24h_display)


if __name__ == "__main__":
    print("=" * 60)
    print("Paperless RAG Chatbot - Monitoring Dashboard")
    print("=" * 60)
    print(f"API Server: {API_URL}")
    print("Make sure the API server is running first!")
    print()
    print("Starting web interface...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
