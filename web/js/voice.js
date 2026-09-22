/* Mic recording -> transcription, and text -> spoken audio. Pure capability, no UI and no
 * knowledge of any particular page -- Learn wires this into its ask panel and its
 * read-aloud control; nothing here is Learn-specific.
 *
 * Both ends are optional server-side (`pip install -e '.[speech]'`); `checkAvailable()`
 * asks once so a caller can decide whether to show a mic or a speaker control at all,
 * rather than showing one that always fails. */

let available = { stt: false, tts: false };

export async function checkAvailable() {
  try {
    available = await (await fetch("/api/speech/status")).json();
  } catch {
    available = { stt: false, tts: false };
  }
  return available;
}

export function sttAvailable() { return available.stt; }
export function ttsAvailable() { return available.tts; }

/* ── recording ────────────────────────────────────────────────────────────────────── */

/* No mimeType is forced: browsers differ in what MediaRecorder can produce (Chrome
 * defaults to webm/opus, Safari to mp4/aac) and the server decodes whatever container
 * arrives via PyAV rather than expecting one specific format. */
export async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const chunks = [];
  const recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const done = new Promise((resolve) => {
    recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
  });
  recorder.start();
  return { recorder, stream, done };
}

/* Stops recording and releases the mic (the browser's recording indicator only clears
 * once every track is stopped, not merely on recorder.stop()); resolves to the clip. */
export async function stopRecording(handle) {
  handle.recorder.stop();
  handle.stream.getTracks().forEach((t) => t.stop());
  return handle.done;
}

/* Recording abandoned without wanting the clip -- the panel it belonged to closed, say.
 * Same track-stopping as stopRecording, without waiting on or keeping the audio. */
export function abortRecording(handle) {
  if (!handle) return;
  try { handle.recorder.stop(); } catch { /* already stopped */ }
  handle.stream.getTracks().forEach((t) => t.stop());
}

export async function transcribe(blob) {
  const form = new FormData();
  form.append("audio", blob, "clip.webm");
  const res = await fetch("/api/speech/transcribe", { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status}`);
  return data.text || "";
}

/* ── speaking ─────────────────────────────────────────────────────────────────────── */

let currentAudio = null;
let currentUrl = null;

export async function fetchSpeech(text, voice) {
  const res = await fetch("/api/speech/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(voice ? { text, voice } : { text }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `${res.status}`);
  }
  return res.blob();
}

/* Plays one clip; `done` resolves when it finishes, fails, or is interrupted by the next
 * playBlob/stopPlayback call, so a sequential reader can simply await it between sentences. */
export function playBlob(blob) {
  stopPlayback();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  currentAudio = audio;
  currentUrl = url;
  const done = new Promise((resolve) => {
    audio.onended = resolve;
    audio.onerror = resolve;
  });
  audio.play().catch(() => {});
  return { audio, done };
}

export function stopPlayback() {
  if (currentAudio) { currentAudio.pause(); currentAudio.onended = null; currentAudio.onerror = null; }
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentAudio = null;
  currentUrl = null;
}
