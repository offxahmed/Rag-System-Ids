import gradio as gr
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from groq import Groq
import json
from datetime import datetime
import io
import base64
from docx import Document
import nltk
from collections import Counter

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Global storage for documents and chat history
document_store = {
    'texts': [],
    'embeddings': [],
    'metadata': [],
    'pdf_previews': []
}
chat_history = []
query_analytics = {
    'total_queries': 0,
    'queries': [],
    'popular_topics': []
}

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with page numbers"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages_text = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                pages_text.append({
                    'text': text,
                    'page': page_num,
                    'filename': pdf_file.name
                })
        
        return pages_text, len(pdf_reader.pages)
    except Exception as e:
        return [], 0

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_file)
        full_text = []
        
        for para_num, paragraph in enumerate(doc.paragraphs, 1):
            text = paragraph.text
            if text.strip():
                full_text.append(text)
        
        # Combine into pages (approximate)
        combined_text = '\n'.join(full_text)
        words_per_page = 300
        words = combined_text.split()
        
        pages_text = []
        for i in range(0, len(words), words_per_page):
            page_text = ' '.join(words[i:i + words_per_page])
            if page_text.strip():
                pages_text.append({
                    'text': page_text,
                    'page': i // words_per_page + 1,
                    'filename': docx_file.name
                })
        
        return pages_text, len(pages_text)
    except Exception as e:
        return [], 0

def create_document_summary(pdf_file, pages_text):
    """Generate AI summary of the PDF document"""
    try:
        sample_text = " ".join([p['text'] for p in pages_text[:3]])[:3000]
        
        prompt = f"""Provide a concise 3-4 sentence summary of this document:

{sample_text}

Summary:"""
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise document summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

def generate_question_suggestions(pdf_file, pages_text):
    """Generate smart question suggestions based on document content"""
    try:
        sample_text = " ".join([p['text'] for p in pages_text[:2]])[:2000]
        
        prompt = f"""Based on this document excerpt, suggest 5 relevant questions a user might want to ask. Format as a numbered list.

{sample_text}

Questions:"""
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests relevant questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return "Could not generate suggestions"

def create_document_statistics(pdf_file, pages_text):
    """Generate statistics about the document"""
    total_pages = len(pages_text)
    total_text = " ".join([p['text'] for p in pages_text])
    word_count = len(total_text.split())
    char_count = len(total_text)
    avg_words_per_page = word_count // total_pages if total_pages > 0 else 0
    estimated_reading_time = word_count // 200
    
    stats = f"""*Document Statistics:*
- Pages: {total_pages}
- Words: {word_count:,}
- Average words per page: {avg_words_per_page}
- Estimated reading time: {estimated_reading_time} minutes"""
    
    return stats

def track_query(query):
    """Track user queries for analytics"""
    global query_analytics
    
    query_analytics['total_queries'] += 1
    query_analytics['queries'].append({
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'length': len(query.split())
    })
    
    if len(query_analytics['queries']) > 100:
        query_analytics['queries'] = query_analytics['queries'][-100:]

def get_analytics_dashboard():
    """Generate analytics dashboard"""
    if query_analytics['total_queries'] == 0:
        return "📊 No queries yet. Start asking questions!"
    
    queries = query_analytics['queries']
    avg_query_length = sum(q['length'] for q in queries) / len(queries) if queries else 0
    
    dashboard = f"""*Analytics Dashboard:*
- Total Queries: {query_analytics['total_queries']}
- Average Query Length: {avg_query_length:.1f} words"""
    
    return dashboard

def chunk_text(pages_data, chunk_size=500, overlap=50):
    """Split text into chunks with sentence boundaries using NLTK"""
    chunks = []
    
    for page_data in pages_data:
        text = page_data['text']
        page_num = page_data['page']
        filename = page_data['filename']
        
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split('.')
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'page': page_num,
                        'filename': filename
                    })
                
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + sentence_words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(sentence_words)
                current_length += sentence_length
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'filename': filename
                })
    
    return chunks

def process_pdfs(pdf_files):
    """Process uploaded PDF and DOCX files"""
    global document_store
    
    if not pdf_files:
        return "⚠ Please upload at least one file.", "", "", ""
    
    document_store = {
        'texts': [],
        'embeddings': [],
        'metadata': [],
        'pdf_previews': []
    }
    
    statistics_text = ""
    suggestions_text = ""
    analytics_text = get_analytics_dashboard()
    total_pages = 0
    total_chunks = 0
    
    for file in pdf_files:
        if file.name.lower().endswith('.pdf'):
            pages_data, num_pages = extract_text_from_pdf(file)
        elif file.name.lower().endswith('.docx'):
            pages_data, num_pages = extract_text_from_docx(file)
        else:
            continue
        
        total_pages += num_pages
        
        if not pages_data:
            continue
        
        stats = create_document_statistics(file, pages_data)
        statistics_text += f"\n\n*{file.name}*\n{stats}"
        
        if file == pdf_files[0]:
            questions = generate_question_suggestions(file, pages_data)
            suggestions_text += f"{questions}"
        
        chunks = chunk_text(pages_data)
        total_chunks += len(chunks)
        
        for chunk in chunks:
            document_store['texts'].append(chunk['text'])
            document_store['metadata'].append({
                'page': chunk['page'],
                'filename': chunk['filename']
            })
    
    if document_store['texts']:
        document_store['embeddings'] = embedding_model.encode(
            document_store['texts'],
            show_progress_bar=False
        )
        
        status = f"""✓ Processing Complete
{len(pdf_files)} files • {total_pages} pages • {total_chunks} chunks"""
        
        return status, statistics_text, suggestions_text, analytics_text
    else:
        return "❌ No text could be extracted", "", "", ""

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve top-k relevant chunks using cosine similarity"""
    if not document_store['texts']:
        return []
    
    query_embedding = embedding_model.encode([query])[0]
    
    similarities = cosine_similarity(
        [query_embedding],
        document_store['embeddings']
    )[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': document_store['texts'][idx],
            'metadata': document_store['metadata'][idx],
            'score': float(similarities[idx])
        })
    
    return results

def generate_answer(query, context_chunks, model_name, temperature):
    """Generate answer using Groq LLM with dynamic parameters"""
    # Model Mapping
    model_map = {
        "Llama-3.3-70b": "llama-3.3-70b-versatile"
    }
    actual_model = model_map.get(model_name, "llama-3.3-70b-versatile")

    context = "\n\n".join([
        f"[From {chunk['metadata']['filename']}, Page {chunk['metadata']['page']}]:\n{chunk['text']}"
        for chunk in context_chunks
    ])
    
    history_context = ""
    if len(chat_history) > 0:
        recent_history = chat_history[-3:]
        history_context = "\n".join([
            f"Previous Q: {h['query']}\nPrevious A: {h['answer'][:200]}..."
            for h in recent_history
        ])
    
    prompt = f"""Based on the following document excerpts, answer the user's question.

Previous Conversation:
{history_context}

Document Context:
{context}

Question: {query}

Provide a clear, concise answer based on the documents. Do not include citations or references to specific pages in your text. If the information isn't in the documents, say so."""

    try:
        response = groq_client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Do not mention source filenames or page numbers in your answer, as these are automatically added by the system."},
                {"role": "user", "content": prompt}
            ],
            temperature=float(temperature),
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def chat(query, history, model_name, temperature):
    """Main chat function with memory and dynamic parameters"""
    if not document_store['texts']:
        return "⚠ Please upload documents first before asking questions."
    
    if not query.strip():
        return "⚠ Please enter a question."
    
    track_query(query)
    
    relevant_chunks = retrieve_relevant_chunks(query, top_k=3)
    
    if not relevant_chunks:
        response = "I couldn't find relevant information in the uploaded documents."
    else:
        answer = generate_answer(query, relevant_chunks, model_name, temperature)
        
        sources = "\n\n*🔎 Sources:*\n"
        for chunk in relevant_chunks:
            filename = os.path.basename(chunk['metadata']['filename'])
            sources += f"- {filename}, Page {chunk['metadata']['page']} (Relevance: {chunk['score']:.2%})\n"
        
        response = answer + sources
    
    chat_history.append({
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': response
    })
    
    return response

def export_chat_history():
    """Export chat history as a formatted TXT string"""
    if not chat_history:
        return None
    
    txt_content = "=== RAG CHAT HISTORY ===\n"
    txt_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    txt_content += "-------------------------------\n\n"
    
    for h in chat_history:
        txt_content += f"User: {h['query']}\n"
        txt_content += f"Assistant: {h['answer']}\n"
        txt_content += "-------------------------------\n\n"
    
    txt_content += "=== END OF HISTORY ===\n"
    
    return txt_content

def clear_all():
    """Clear all data"""
    global document_store, chat_history, query_analytics
    document_store = {
        'texts': [],
        'embeddings': [],
        'metadata': [],
        'pdf_previews': []
    }
    chat_history = []
    query_analytics = {
        'total_queries': 0,
        'queries': [],
        'popular_topics': []
    }
    return "🔄 All data cleared", "", "", "📊 No queries yet", [], ""

# Professional CSS with animations and modern theme
custom_css = """
:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --sidebar-width: 300px;
}

.gradio-container {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Navbar Styling */
.navbar {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 1000;
    margin-bottom: 2rem;
    border-radius: 0 0 15px 15px;
}

.navbar-brand {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sidebar Enhancements */
.sidebar {
    background: var(--card-bg) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 1.5rem !important;
}

/* Button Styling */
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4) !important;
}

/* Chatbot Styling */
.chatbot-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(30, 41, 59, 0.5) !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message-user, .message-bot {
    animation: fadeIn 0.4s ease-out forwards;
}

/* Header Visibility and Typography */
h1, h2, h3, .gr-markdown, .gr-markdown p, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--text-main) !important;
}

.sidebar h2 {
    color: var(--primary) !important;
    font-size: 1.25rem !important;
    margin-top: 1rem !important;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.5) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

/* Hide Gradio Branding */
footer { display: none !important; }
"""

with gr.Blocks(title="Intelligent Document Search") as demo:
    
    # Navbar
    with gr.Row(elem_classes=["navbar"]):
        gr.HTML("""
            <div class="navbar-brand">🚀 RAG-IDS Professional</div>
            <div style="display: flex; gap: 20px; align-items: center;">
                <span style="color: #94a3b8;">v2.0 Stable</span>
                <div style="width: 10px; height: 10px; background: #22c55e; border-radius: 50%; box-shadow: 0 0 10px #22c55e;"></div>
            </div>
        """)

    with gr.Sidebar(elem_classes=["sidebar"]) as sidebar:
        gr.Markdown("## 📂 Workspace")
        pdf_input = gr.File(label="Upload PDF or DOCX", file_count="multiple", file_types=[".pdf", ".docx"])
        process_btn = gr.Button("Analyze Documents", variant="primary")
        status_output = gr.Textbox(label="System Status", value="Ready", interactive=False)
        
        gr.Markdown("---")
        gr.Markdown("## ⚙ Configuration")
        model_dropdown = gr.Dropdown(
            choices=["Llama-3.3-70b"],
            value="Llama-3.3-70b",
            label="Neural Model"
        )
        temp_slider = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.1, label="Creativity (Temp)")
        
        gr.Markdown("---")
        gr.Markdown("## 🛠 Actions")
        with gr.Row():
            clear_chat_btn = gr.Button("🗑 Clear", size="sm")
            export_btn = gr.Button("📤 Export", size="sm")
        reset_btn = gr.Button("🔄 Factory Reset", size="sm", variant="stop")
        download_output = gr.File(label="Exported File", visible=False)

    with gr.Row():
        # Main Column - Chat Interface
        with gr.Column(scale=3):
            gr.Markdown("### 💬 Intelligence Hub")
            chatbot = gr.Chatbot(
                label="Conversation Window", 
                height=650, 
                show_label=False,
                elem_classes=["chatbot-container"]
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask anything about the uploaded knowledge base...", 
                    scale=7, 
                    show_label=False,
                    container=False
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")

        # Right Column - Side Info
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("### 📊 Doc Insights")
                stats_output = gr.Markdown("Awaiting data...")
            
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("### 💡 AI Suggestions")
                suggestions_output = gr.Markdown("Upload docs to generate...")
            
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("### 📈 Usage Analytics")
                analytics_output = gr.Markdown(get_analytics_dashboard())

    # Footer
    gr.Markdown("<center style='color: #64748b; font-size: 0.8rem; margin-top: 2rem;'>© 2025 RAG-IDS. All rights reserved.</center>")

    # Event Handlers

    def process_wrapper(files):
        if not files:
            return "⚠ Please upload at least one document", "", "", ""
        
        return process_pdfs(files)

    process_btn.click(
        fn=process_wrapper,
        inputs=[pdf_input],
        outputs=[status_output, stats_output, suggestions_output, analytics_output]
    )
    
    def respond(message, history, model, temp):
        if not message:
            return history
        bot_message = chat(message, history, model, temp)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_message})
        return history
    
    msg.submit(respond, [msg, chatbot, model_dropdown, temp_slider], chatbot).then(
        lambda: "", None, msg
    ).then(
        get_analytics_dashboard, None, analytics_output
    )
    
    submit_btn.click(respond, [msg, chatbot, model_dropdown, temp_slider], chatbot).then(
        lambda: "", None, msg
    ).then(
        get_analytics_dashboard, None, analytics_output
    )
    
    clear_chat_btn.click(lambda: [], None, chatbot)
    
    def export_wrapper():
        content = export_chat_history()
        if content:
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return gr.File(value=filename, visible=True)
        return gr.File(visible=False)
    
    export_btn.click(fn=export_wrapper, inputs=[], outputs=[download_output])
    
    reset_btn.click(
        fn=clear_all,
        outputs=[status_output, stats_output, suggestions_output, analytics_output, chatbot, msg]
    )

# Launch
if _name_ == "_main_":
    demo.launch(
        share=False,
        show_error=True,
        css=custom_css
    )