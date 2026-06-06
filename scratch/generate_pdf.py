import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header
        self.drawString(54, 750, "PodcastBot — Video RAG Q&A Solution Writeup")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 742, 612 - 54, 742)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 40, page_text)
        self.drawString(54, 40, "GitHub Repository: https://github.com/molubhai08/Video_Rag")
        self.line(54, 52, 612 - 54, 52)
        
        self.restoreState()

def build_pdf(filename="solution_writeup.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1e1b4b")  # Dark indigo
    accent_color = colors.HexColor("#4f46e5")   # Royal indigo
    text_color = colors.HexColor("#1e293b")     # Slate 800
    bg_light = colors.HexColor("#f8fafc")       # Slate 50
    border_color = colors.HexColor("#e2e8f0")   # Slate 200
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'H1Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=accent_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Video RAG Q&A Bot — Solution Writeup", title_style))
    story.append(Spacer(1, 10))
    
    metadata_text = """
    <b>Project Code Repository:</b> <font color="#4f46e5"><a href="https://github.com/molubhai08/Video_Rag">https://github.com/molubhai08/Video_Rag</a></font><br/>
    <b>Demo Video Link:</b> <font color="#4f46e5"><a href="https://drive.google.com/file/d/1xJhp6rtrXSHlRKektrkxna3-oFBTugUL/view?usp=sharing">Google Drive Video</a></font><br/>
    <b>Deployment Note:</b> Not currently live on cloud due to YouTube IP blocks on cloud instances (details below). Runs locally.<br/>
    """
    p_meta = Paragraph(metadata_text, body_style)
    t_meta = Table([[p_meta]], colWidths=[504])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('PADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1. The Workflow, Prompts, and Steps", h1_style))
    story.append(Paragraph("The system implements a <b>Two-Level Retrieval-Augmented Generation (RAG)</b> pipeline to pinpoint answers and timestamps from long video transcripts without requiring database-backed vector store infrastructure.", body_style))
    
    steps = [
        "<b>1. Extraction:</b> The user inputs a YouTube URL. The system extracts the 11-character video ID.",
        "<b>2. Transcript Processing & Local Caching:</b> Checks local JSON cache first. If it's a miss, fetches the transcript via the <code>youtube-transcript-api</code> client, groups elements into semantic chunks of ~450 words (with 100 words overlap), and writes to local cache.",
        "<b>3. Level 1 Retrieval:</b> Fits a lightweight <code>TfidfVectorizer</code> (unigrams + bigrams) on the fly over chunks, computes cosine similarity against the query vector, and retrieves the top 3 chunks.",
        "<b>4. Level 2 Refinement:</b> Scores individual snippets inside the best-matching chunk using keyword word-overlaps to pick the exact starting seconds timestamp.",
        "<b>5. Answer Generation:</b> Sends the user query + top 3 context chunks to the Groq API (using the <code>llama-3.1-8b-instant</code> model).",
        "<b>6. Seeking Interface:</b> Displays the response in the UI. When the user clicks the timestamp link, the app triggers HTML5 <code>postMessage</code> script calls to automatically seek the YouTube iframe player."
    ]
    for step in steps:
        story.append(Paragraph(f"&bull; {step}", bullet_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("System Prompt Context for Groq LLM", h2_style))
    sys_prompt = "You are an expert podcast analyst. Answer the user's question based ONLY on the provided transcript segments. Be concise and insightful (2-4 sentences). If the answer isn't in the transcript, say so honestly. Do NOT make up information."
    story.append(Table([[Paragraph(f"<b>System Prompt:</b><br/>{sys_prompt}", code_style)]], colWidths=[504], style=[
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
    ]))
    
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("2. Tools & Models Used", h1_style))
    tools = [
        "<b>Frontend:</b> Vanilla HTML5, CSS3, & Modern Javascript (Inter & Space Grotesk typography, glassmorphism, custom interactive controls).",
        "<b>Backend Framework:</b> Flask (Python 3.10+) for APIs, thread execution, and caching.",
        "<b>Transcript Parsing:</b> <code>youtube-transcript-api</code> configured with custom cookie headers to bypass basic scraping filters.",
        "<b>Retrieval Engine:</b> Scikit-learn (TF-IDF vectorizer + Cosine Similarity computations).",
        "<b>LLM Engine:</b> Groq API with <code>llama-3.1-8b-instant</code> for sub-300ms response latencies."
    ]
    for tool in tools:
        story.append(Paragraph(f"&bull; {tool}", bullet_style))
        
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("3. Accuracy Check & Verification", h1_style))
    checks = [
        "<b>Grounding Audit:</b> Checked responses to ensure no hallucinations occur. The Groq model returns negative responses correctly when the transcript does not contain the answer.",
        "<b>Timestamp Pinpointing:</b> Validated against Lex Fridman and Elon Musk interviews. Verified generated timestamps match ground-truth timestamps within 5 seconds.",
        "<b>Cache Testing:</b> Confirmed that subsequent questions on cached video IDs load immediately (< 10ms) from local memory rather than hitting YouTube rates."
    ]
    for check in checks:
        story.append(Paragraph(f"&bull; {check}", bullet_style))
        
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("4. Current Limitations & Future Improvements", h1_style))
    
    story.append(Paragraph("A. TF-IDF for Semantic Search", h2_style))
    story.append(Paragraph("<b>Limitation:</b> We utilized TF-IDF because it is lightweight, requires no heavy external machine learning models/downloads, and runs in milliseconds on standard CPU units. However, it relies entirely on keyword matching. If the user asks a question using synonyms instead of exact words used in the video, TF-IDF might fail to rank the correct chunk highest.<br/><b>Improvement:</b> In future versions, we will transition to dense semantic vector embeddings (e.g. OpenAI's <code>text-embedding-3-small</code> or Hugging Face's <code>BGE-m3</code>) to match conceptually.", body_style))
    
    story.append(Paragraph("B. Local JSON Caching", h2_style))
    story.append(Paragraph("<b>Limitation:</b> Transcripts are cached as local <code>.json</code> files, and active tasks are tracked using in-memory Python dictionaries with thread locks. This does not scale horizontally across multi-worker cloud servers because worker processes do not share memory and local disk storage is ephemeral.<br/><b>Improvement:</b> We will integrate <b>Redis</b> for task status and transcript caching to support highly available, horizontally scaled cloud configurations.", body_style))

    story.append(Paragraph("C. Cloud Rate Limits (YouTube Blocks)", h2_style))
    story.append(Paragraph("<b>Limitation:</b> Deployment to cloud services (like Azure App Service or AWS) throws errors because YouTube heavily rate-limits/blocks datacenter IP blocks.<br/><b>Improvement:</b> Implement rotating residential proxy configuration options (e.g. Webshare integration) inside the YouTube client HTTP agent.", body_style))
    
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("D. AI Multimodal Blindness (Text-Only Limitation)", h2_style))
    story.append(Paragraph("<b>Limitation:</b> The AI model struggles with visual reference questions because it performs search on text transcripts only. If a host points to a slide or prototype and says, <i>'As you can see here, this is the main issue'</i>, the text-only RAG misses the visual context of what <i>'this'</i> refers to.<br/><b>Improvement:</b> In future versions, we can sample video frames (e.g., every 5 seconds) and pass them to a Vision-Language Model (VLM like Gemini Flash) to append visual summaries to the text transcripts.", body_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)

if __name__ == "__main__":
    build_pdf()
    print("PDF Generated successfully with only Multimodal struggle.")
