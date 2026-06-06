# PodcastBot — Video RAG Q&A App

A premium, YouTube-style web application that performs Retrieval-Augmented Generation (RAG) on YouTube video transcripts. Paste a YouTube URL, index the transcript, and ask questions. The app answers using Groq's Llama 3.1 model and can automatically seek the YouTube video player to the exact relevant timestamp.

---

## Features
- **YouTube Transcript Processing**: Automatically downloads and processes English transcripts.
- **Two-Level Retrieval**: Semantic chunking + TF-IDF index (Level 1) combined with keyword overlap alignment (Level 2) to pinpoint exact timestamps.
- **Interactive YouTube Player**: Synchronized with Q&A timestamps (click a timestamp to jump in the video).
- **Premium YouTube-style UI**: Glassmorphic dark mode layout featuring a main video player area and live-chat style Q&A sidebar.

---

## Local Setup

### 1. Prerequisites
- Python 3.10+
- A Groq API Key (Get a free key at [Groq Console](https://console.groq.com/))

### 2. Installation
Clone/download the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_custom_secret_key_here
```

### 4. Running the App

#### Development Mode (Flask Dev Server)
```bash
python app.py
```
Go to `http://127.0.0.1:5000` in your web browser.

#### Production Mode (Gunicorn - Unix/Linux)
```bash
gunicorn --workers=1 --threads=4 --timeout 120 app:app
```

---

## Azure Deployment (Azure App Service)

This application is ready for deployment on **Azure App Service** (Linux, Python 3.x).

### Step 1: Create the Web App in Azure Portal
1. Go to the [Azure Portal](https://portal.azure.com/).
2. Click **Create a resource** -> **Web App**.
3. Configure the following:
   - **Publish**: Code
   - **Runtime stack**: Python 3.10 or Python 3.11
   - **Operating System**: Linux
   - **Pricing Plan**: Free (F1) for testing, or Basic (B1) for continuous availability.

### Step 2: Configure Environment Variables
1. Navigate to your Web App resource in the Azure Portal.
2. Under **Settings** (left sidebar), click **Configuration** (or **Environment Variables** in newer UI).
3. Add a new Application Setting:
   - **Name**: `GROQ_API_KEY`
   - **Value**: *[Your Groq API Key]*
4. Add another setting:
   - **Name**: `FLASK_SECRET_KEY`
   - **Value**: *[Any random secure string]*
5. Click **Save** at the top.

### Step 3: Configure the Startup Command
1. In the Web App menu, go to **Settings** -> **Configuration** -> **General settings**.
2. Locate the **Startup Command** field.
3. Enter the following to execute our custom startup script:
   ```bash
   bash startup.sh
   ```
4. Click **Save**.

### Step 4: Deploy the Code
You can deploy your code using the **Azure Resources extension** in VS Code:
1. Install the **Azure Resources** and **Azure App Service** extensions in VS Code.
2. Sign in to your Azure Account.
3. In the Azure sidebar tab, expand your subscription and locate your Web App.
4. Right-click the Web App name and select **Deploy to Web App...**.
5. Select the `video_rag` root folder and confirm.
6. Once deployment finishes, your app will be live at `https://<your-app-name>.azurewebsites.net`!
