#!/usr/bin/env python3
"""
Fast Web interface for the Paperless RAG Chatbot using Gradio.

Optimized for sub-2-second responses on Apple Silicon.
"""
import gradio as gr
import requests
import time

API_URL = "http://localhost:8001"


def query_chatbot(message, history):
    """Query the fast RAG chatbot."""
    try:
        start_time = time.time()
        
        # Detect analytical queries for more context
        analytical_keywords = [
            'alle', 'gesamt', 'übersicht', 'tabelle', 'liste',
            'all', 'total', 'overview', 'table', 'list'
        ]
        is_analytical = any(kw in message.lower() for kw in analytical_keywords)
        
        # Fewer results for speed (3 default, 6 for analytical)
        n_results = 6 if is_analytical else 3
        
        response = requests.post(
            f"{API_URL}/query",
            json={"question": message, "n_results": n_results},
            timeout=30  # 30s is enough for fast mode
        )
        
        latency = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result['answer']
            
            # Add sources with clickable links
            if result['sources']:
                sources_text = "\n\n**Sources:**\n"
                for i, source in enumerate(result['sources'][:3], 1):
                    score = source.get('score', 0) * 100
                    title = source.get('title', 'Unknown')
                    url = source.get('url', '')
                    
                    if url:
                        sources_text += f"{i}. **{title}** ({score:.0f}%)\n   {url}\n"
                    else:
                        sources_text += f"{i}. {title} ({score:.0f}%)\n"
                
                answer += sources_text
            
            # Add latency info
            server_latency = result.get('latency_ms', 0)
            answer += f"\n\n*Response time: {latency:.1f}s (server: {server_latency}ms)*"
            
            return answer
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.Timeout:
        return "Request timed out. Check if Ollama and the API server are running."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to API server. Run 'make serve-api' first."
    except Exception as e:
        return f"Error: {str(e)}"


def check_health():
    """Check server health."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            collection = data.get('collection', {})
            config = data.get('config', {})
            
            return (
                f"Status: Healthy\n"
                f"Chunks: {collection.get('points_count', 0)}\n"
                f"LLM: {config.get('llm_model', 'unknown')}\n"
                f"Embeddings: {config.get('embedding_model', 'unknown')}"
            )
        else:
            return "Server has issues"
    except requests.exceptions.ConnectionError:
        return "Server not running.\nRun: make serve-api"
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Paperless RAG Chatbot") as demo:
    gr.Markdown("# Paperless RAG Chatbot (Fast Mode)")
    gr.Markdown("Ask questions about your Paperless-NGX documents. Responses in ~1-2 seconds.")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                query_chatbot,
                examples=[
                    "Was steht in meiner letzten Rechnung?",
                    "Find all documents from Amazon",
                    "Zeige mir Dokumente aus 2024",
                    "What payments did I make this year?",
                    "Liste alle Verträge auf"
                ]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Status")
            health_output = gr.Textbox(
                label="Server Health",
                interactive=False,
                lines=4
            )
            health_btn = gr.Button("Refresh")
            health_btn.click(check_health, outputs=health_output)
            demo.load(check_health, outputs=health_output)
            
            gr.Markdown("---")
            gr.Markdown("### Quick Links")
            gr.Markdown("[API Docs](http://localhost:8001/docs)")
            gr.Markdown("[Debug Search](http://localhost:8001/debug/search/test)")


if __name__ == "__main__":
    print("=" * 50)
    print("Paperless RAG Chatbot - Fast Web Interface")
    print("=" * 50)
    print(f"API Server: {API_URL}")
    print("Make sure the API is running: make serve-api")
    print()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
