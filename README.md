# Movie Metadata RAG Assistant

A Flask web application that answers movie-related questions using
**Retrieval-Augmented Generation (RAG)**:

1. **Retrieve** — TF-IDF + cosine similarity over the
   [`movies_metadata.csv`](https://hiperc.buffalostate.edu/courses/movies_metadata.csv)
   dataset (top 5 rows).
2. **Build context** — A compact, structured block of the retrieved rows.
3. **Ground the LLM** — Send the question + context to a locally hosted
   [Ollama](https://ollama.com/) model
   (`minimax-m2.1:cloud`) at `http://localhost:11434/api/generate`.
4. **Render** — Show both the retrieved rows and the grounded answer in the UI.

```
Question  ->  Retrieve (TF-IDF)  ->  Build Context  ->  LLM Answer (Ollama)
```

> **Repository / folder name:** `LastName_Movie_metadata`
> Replace `LastName` with your actual last name before submitting (e.g. `Smith_Movie_metadata`).

---

## Project structure

```
LastName_Movie_metadata/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

`app.py` implements:

| Function                              | Purpose                                                        |
| ------------------------------------- | -------------------------------------------------------------- |
| `load_movies()`                       | Download/cache the CSV, keep useful columns, clean rows.       |
| `retrieve_movies(question, top_k=5)`  | Return the top-k rows + cosine similarity scores.              |
| `build_context(rows)`                 | Produce a compact, structured context block for the LLM.       |
| `ask_ollama(question, context)`       | POST a grounded prompt to `/api/generate` (`stream: false`).   |
| Flask route `/`                       | Form input, retrieval results, and final answer rendering.    |

The LLM prompt explicitly instructs the model to answer **only** from the
retrieved context and to say so clearly when the context is insufficient
(no hallucinations).

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

Install Ollama from <https://ollama.com/download>, then verify and pull the
required model:

```bash
ollama --version
ollama run minimax-m2.1:cloud
```

Keep the Ollama service active in another terminal while running Flask.
The app expects the API at `http://localhost:11434/api/generate`.

### 3. Run the Flask app

```bash
python app.py
```

Open the app at:

```
http://127.0.0.1:5005
```

---

## Usage

Type a movie-related question and press **Ask**. The page will display:

- **Retrieved Movies** — the top 5 rows from the CSV with similarity scores,
  title, genres, release date, rating, and overview.
- **Grounded Answer** — the LLM's response, generated using *only* the
  retrieved rows as context.
- A collapsible **context preview** showing exactly what was sent to the LLM.

### Example questions

- "Find movies about space travel and exploration."
- "What are some high-rated drama movies in this dataset?"
- "Which retrieved movies are related to war themes?"
- "Find movies with romance and comedy elements."

### Sample query & output

**Question:** *Find movies about space travel and exploration.*

**Top retrieved (excerpt):**

| # | Score  | Title              | Genres                   | Rating |
|---|--------|--------------------|--------------------------|--------|
| 1 | 0.41xx | Apollo 13          | Drama, Adventure         | 7.6    |
| 2 | 0.38xx | Interstellar       | Adventure, Drama, Sci-Fi | 8.3    |
| 3 | 0.34xx | 2001: A Space Odyssey | Mystery, Sci-Fi      | 8.1    |
| ... | ...  | ...                | ...                      | ...    |

**Grounded answer (excerpt):**

> Based on the retrieved movies, *Apollo 13* dramatizes a real NASA lunar
> mission and *Interstellar* follows astronauts traveling through a wormhole
> to find a habitable planet. *2001: A Space Odyssey* explores deep-space
> travel and contact with an alien intelligence...

> Replace this section with your own screenshot — `screenshot.png` — once
> you run the app locally.

---

## Error handling

- **Empty question** → friendly inline error, no LLM call made.
- **Ollama unreachable** → clear message instructing user to start `ollama serve`
  and confirm the model is pulled.
- **Ollama HTTP error / timeout / invalid JSON** → surfaces a readable error
  and still shows the retrieved rows.
- **No matching rows** → tells the user nothing matched and skips the LLM call.

---

## Submission checklist (per the rubric)

- [ ] Repository named `LastName_Movie_metadata` (rename folder + repo).
- [ ] `app.py`, `templates/index.html`, `requirements.txt`, `README.md` present.
- [ ] App runs on **port 5005**.
- [ ] CSV loaded and cleaned correctly.
- [ ] TF-IDF + cosine similarity retrieval, top 5.
- [ ] Ollama call uses `minimax-m2.1:cloud` with a grounded prompt.
- [ ] UI shows retrieved rows **and** generated answer.
- [ ] Screenshot of running app committed (e.g. `screenshot.png`).
- [ ] GitHub URL submitted via the class Google Form Dropbox.
