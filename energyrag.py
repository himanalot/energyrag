import os
import glob
import requests
from torch.utils.tensorboard import SummaryWriter
import gradio as gr
import io
from PyPDF2 import PdfReader
import time

# Global placeholder for GUI-only usage; actual docs come from upload
docs = []

# Note: This script requires PyTorch, HuggingFace Transformers, FAISS, and SentenceTransformers.
# Install them with:
# pip install torch transformers sentence-transformers faiss-cpu

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    raise ImportError("PyTorch is not installed. Please install it with 'pip install torch'.")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss

# 1. Prepare retrieval index
class Retriever:
    def __init__(self, corpus_texts=None, embedder_model="all-MiniLM-L6-v2", index_path=None):
        if not corpus_texts:
            raise ValueError("No documents found in corpus; please add text files to the 'corpus/' directory.")
        if index_path and os.path.exists(index_path):
            self.embedder = SentenceTransformer(embedder_model)
            self.index = faiss.read_index(index_path)
            self.texts = corpus_texts or []
            return
        self.embedder = SentenceTransformer(embedder_model)
        embeddings = self.embedder.encode(corpus_texts, convert_to_tensor=True)
        # Handle single-document case: ensure embeddings is 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.cpu().numpy())
        self.texts = corpus_texts
        if index_path:
            faiss.write_index(self.index, index_path)

    def retrieve(self, query, top_k=5):
        q_emb = self.embedder.encode([query], convert_to_tensor=True).cpu().numpy()
        scores, idxs = self.index.search(q_emb, top_k)
        return [(self.texts[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

# 2. Energy-Based Generator
class ERAGModel(torch.nn.Module):
    def __init__(self, generator_name="t5-small", energy_dim=512, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            raise ValueError("embed_dim must be provided to match the retriever embedding dimension")
        self.embed_dim = embed_dim
        self.tokenizer = AutoTokenizer.from_pretrained(generator_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_name)
        # energy head: input dim = generator hidden + embedding dim
        hidden_size = self.generator.config.d_model
        energy_input_dim = hidden_size + self.embed_dim
        self.energy_head = torch.nn.Sequential(
            torch.nn.Linear(energy_input_dim, energy_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(energy_dim, 1)
        )

    def forward(self, input_ids, attention_mask, retrieved_emb):
        # generate hidden states
        outputs = self.generator.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # use pooled encoder state
        enc_state = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden]
        # ensure retrieved_emb has batch dimension
        if retrieved_emb.ndim == 1:
            retrieved_emb = retrieved_emb.unsqueeze(0)  # [1, embed_dim]
        # ensure batch sizes match
        if retrieved_emb.shape[0] != enc_state.shape[0]:
            raise ValueError(f"Batch size mismatch: enc_state {enc_state.shape} vs retrieved_emb {retrieved_emb.shape}")
        combined = torch.cat([enc_state, retrieved_emb], dim=-1)
        energy = self.energy_head(combined)
        return energy

# 3. Training loop skeleton
def train_e_rag(model, retriever, queries, optimizer, device):
    writer = SummaryWriter(log_dir="runs/e_rag")
    step = 0
    model.train()
    for q in queries:
        # Retrieve positive docs
        docs_scores = retriever.retrieve(q)
        top_doc, _ = docs_scores[0]
        # Tokenize query+doc with QA prefix
        enc = model.tokenizer("question: " + q + "  context: " + top_doc + "  answer in English:", return_tensors="pt", truncation=True, padding=True).to(device)
        # Compute retrieved embedding
        doc_emb = retriever.embedder.encode([top_doc], convert_to_tensor=True).to(device)
        # Positive energy
        e_pos = model(enc.input_ids, enc.attention_mask, doc_emb)
        # Negative: use second-best retrieved doc as a hard negative
        docs_scores_neg = retriever.retrieve(q, top_k=2)
        neg_doc = docs_scores_neg[1][0] if len(docs_scores_neg) > 1 else docs_scores_neg[0][0]
        enc_neg = model.tokenizer("question: " + q + "  context: " + neg_doc + "  answer in English:", return_tensors="pt", truncation=True, padding=True).to(device)
        neg_emb = retriever.embedder.encode([neg_doc], convert_to_tensor=True).to(device)
        e_neg = model(enc_neg.input_ids, enc_neg.attention_mask, neg_emb)
        # Contrastive loss
        loss = torch.relu(e_pos - e_neg + 1.0).mean()
        writer.add_scalar("train/loss", loss.item(), step)
        step += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.close()

# 4. Inference
@torch.no_grad()
def generate_with_energy(model, retriever, query, device, top_k=5):
    model.eval()
    docs = retriever.retrieve(query, top_k)
    best = None
    lowest_energy = float('inf')
    for doc, _ in docs:
        enc = model.tokenizer("question: " + query + "  context: " + doc + "  answer in English:", return_tensors="pt", truncation=True, padding=True).to(device)
        doc_emb = retriever.embedder.encode([doc], convert_to_tensor=True).to(device)
        e = model(enc.input_ids, enc.attention_mask, doc_emb)
        if e.item() < lowest_energy:
            # generate text for best
            gen_ids = model.generator.generate(enc.input_ids, attention_mask=enc.attention_mask)
            best = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            lowest_energy = e.item()
    # Fallback: if generation is too short or repeats the query, use the context sentence directly
    if best is None or len(best.split()) < 3 or best.strip().lower() in query.lower():
        # return the first retrieved context sentence
        best = docs[0][0]
    return best

@torch.no_grad()
def generate_rag(model, retriever, question, device, top_k=5):
    # standard RAG: retrieve top document, then generate directly
    docs = retriever.retrieve(question, top_k)
    context = docs[0][0]
    prompt = "question: " + question + "  context: " + context + "  answer in English:"
    enc = model.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    gen_ids = model.generator.generate(enc.input_ids, attention_mask=enc.attention_mask)
    return model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

# --- Gradio GUI ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ask_all(question, uploaded_file=None):
    # load docs from uploaded file or use SQuAD chunks
    if uploaded_file is not None:
        # Determine file bytes
        if hasattr(uploaded_file, "data"):  # NamedTemporaryFile or NamedString
            file_bytes = uploaded_file.data
        elif isinstance(uploaded_file, str):  # path string
            with open(uploaded_file, "rb") as f:
                file_bytes = f.read()
        else:
            file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        if filename.lower().endswith(".pdf"):
            # extract text from PDF
            reader = PdfReader(io.BytesIO(file_bytes))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = file_bytes.decode("utf-8")
        docs_gui = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    else:
        docs_gui = docs
    # re-instantiate retriever with docs_gui
    retr = Retriever(docs_gui)
    start = time.time()
    rag_ans = generate_rag(model, retr, question, device)
    rag_time = time.time() - start
    start = time.time()
    erag_ans = generate_with_energy(model, retr, question, device)
    erag_time = time.time() - start
    return rag_ans, f"{rag_time:.3f}s", erag_ans, f"{erag_time:.3f}s"

# Initialize device and load model for GUI
# Choose hardware acceleration: MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"[INFO] Using device: {device} for model inference in GUI")

# Create a dummy embedder to get embedding dimension
tmp_embedder = SentenceTransformer("all-MiniLM-L6-v2")
embed_dim = tmp_embedder.get_sentence_embedding_dimension()

model = ERAGModel(embed_dim=embed_dim).to(device)

iface = gr.Interface(
    fn=ask_all,
    inputs=[
        gr.Textbox(lines=2, label="Question"),
        gr.File(label="Upload Document (optional)", file_types=['.txt', '.pdf'])
    ],
    outputs=[
        gr.Textbox(label="Standard RAG Response"),
        gr.Textbox(label="RAG Latency"),
        gr.Textbox(label="Energy-RAG Response"),
        gr.Textbox(label="E-RAG Latency")
    ],
    title="E-RAG vs RAG Demo",
    description="Upload a text file or use SQuAD data. Enter a question to compare standard RAG with E-RAG."
)

if __name__ == "__main__":
    print("[INFO] Launching GUI...")
    iface.launch(share=False)
