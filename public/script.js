// The text chunks of the uploaded files (and their embeddings) live here, in the browser.
// Every question sends them to the server along with the question.
let chunks = [];
let embeddings = [];

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusText = document.getElementById("status");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");

// Add a chat bubble to the screen and return it
function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `message ${who}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

// Read the error message the server sent back
async function errorText(res) {
  try {
    const data = await res.json();
    return data.detail || res.statusText;
  } catch {
    return res.status === 413 ? "File too large (max ~4MB)." : res.statusText;
  }
}

// ---------- Step 1: upload files ----------
uploadBtn.addEventListener("click", async () => {
  if (fileInput.files.length === 0) {
    statusText.textContent = "Pick a file first.";
    return;
  }

  const formData = new FormData();
  for (const file of fileInput.files) {
    formData.append("files", file);
  }

  uploadBtn.disabled = true;
  statusText.textContent = "Reading files and creating embeddings...";

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await errorText(res));

    const data = await res.json();
    chunks = data.chunks;
    embeddings = data.embeddings;
    messages.innerHTML = "";
    statusText.textContent = `✅ Ready! ${chunks.length} chunks loaded. Ask a question.`;
  } catch (err) {
    statusText.textContent = `❌ ${err.message}`;
  } finally {
    uploadBtn.disabled = false;
  }
});

// ---------- Step 2: ask questions ----------
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  if (chunks.length === 0) {
    statusText.textContent = "Upload a file and click 'Process files' first.";
    return;
  }

  addMessage(question, "user");
  questionInput.value = "";
  const reply = addMessage("Thinking...", "bot");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, chunks, embeddings }),
    });
    if (!res.ok) throw new Error(await errorText(res));

    const data = await res.json();
    reply.textContent = data.answer;
  } catch (err) {
    reply.textContent = `Error: ${err.message}`;
  }
});
