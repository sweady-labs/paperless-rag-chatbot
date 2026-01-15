#!/usr/bin/env python3
"""
Web interface for the Paperless RAG Chatbot using Gradio
"""
import gradio as gr
import requests
import sys

API_URL = "http://localhost:8001"

def query_chatbot(message, history):
    """Query the RAG chatbot"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": message, "n_results": 3},  # Reduced to 3 for faster responses
            timeout=120  # Increased for complex LLM queries
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['answer']
            
            # Add sources with clickable links
            if result['sources']:
                sources_text = "\n\n**📚 Sources:**\n"
                for i, source in enumerate(result['sources'][:3], 1):
                    # Handle both score formats (simple and hybrid)
                    if 'score' in source:
                        score_percent = source['score'] * 100
                    elif 'hybrid_score' in source:
                        score_percent = source['hybrid_score'] * 100
                    else:
                        score_percent = 0
                    
                    # Add URL if available - Gradio ChatInterface needs explicit URLs
                    if 'url' in source and source['url']:
                        sources_text += f"{i}. **{source['title']}** (relevance: {score_percent:.1f}%)\n"
                        sources_text += f"   🔗 {source['url']}\n"
                    else:
                        sources_text += f"{i}. {source['title']} (relevance: {score_percent:.1f}%)\n"
                answer += sources_text
            
            return answer
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out (2 min). Try a simpler question or check if Ollama is running."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def check_health():
    """Check server health"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            collection = data.get('collection', {})
            return f"✅ Server is healthy\n📊 Indexed chunks: {collection.get('points_count', 'unknown')}"
        else:
            return "⚠️ Server is running but might have issues"
    except:
        return "❌ Cannot connect to server"

# Create Gradio interface
with gr.Blocks(title="Paperless RAG Chatbot") as demo:
    gr.Markdown("# 📄 Paperless RAG Chatbot")
    gr.Markdown("Ask questions about your documents in Paperless-NGX")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                query_chatbot,
                examples=[
                    "What invoices do I have from 2024?",
                    "Summarize my tax documents",
                    "Find contracts related to insurance",
                    "What documents mention payments?"
                ]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### System Status")
            health_output = gr.Textbox(label="Health Check", interactive=False)
            health_btn = gr.Button("🔄 Check Status")
            health_btn.click(check_health, outputs=health_output)
            
            # Show initial health on load
            demo.load(check_health, outputs=health_output)

if __name__ == "__main__":
    print("Starting Gradio web interface...")
    print(f"Make sure the API server is running at {API_URL}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
