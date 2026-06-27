import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Database, Layers, Cpu, Sliders, Settings, Plus, Search, Send,
  TrendingUp, ThumbsUp, ThumbsDown, FileText, Sparkles, ChevronRight,
  Terminal, Activity, ShieldAlert, Key, Play, Network, Upload,
  BarChart2, BookOpen, FlaskConical, Zap, X, CheckCircle2, AlertCircle,
  RefreshCw, ChevronDown, ChevronUp, Eye, Trash2, Copy, Info, GitBranch,
  ArrowUpRight, MessageSquare, Clock, DollarSign, Hash, Star
} from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, RadarChart,
  Radar, PolarGrid, PolarAngleAxis
} from 'recharts';
import './index.css';

const API = 'http://localhost:8082/api/v1';

// ─── API Helpers ─────────────────────────────────────────────────────────────
async function apiFetch(path: string, opts?: RequestInit) {
  try {
    const r = await fetch(`${API}${path}`, opts);
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }
    return r.json();
  } catch (e: any) {
    throw new Error(e.message || 'Network error');
  }
}

function formFetch(path: string, data: Record<string, any>, method = 'POST') {
  const fd = new FormData();
  for (const [k, v] of Object.entries(data)) fd.append(k, String(v));
  return apiFetch(path, { method, body: fd });
}

// ─── Score Color ──────────────────────────────────────────────────────────────
function scoreColor(v: number): string {
  if (v >= 0.85) return 'var(--teal-400)';
  if (v >= 0.65) return 'var(--amber-400)';
  return 'var(--red-400)';
}
function scoreBadge(v: number): string {
  if (v >= 0.85) return 'badge-teal';
  if (v >= 0.65) return 'badge-amber';
  return 'badge-red';
}
function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }

// ─── Score Bar Component ──────────────────────────────────────────────────────
function ScoreBar({ value, label }: { value: number; label: string }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: 11, fontWeight: 700, color: scoreColor(value) }}>{pct(value)}</span>
      </div>
      <div className="score-bar">
        <div className="score-bar-fill" style={{ width: `${value * 100}%`, background: scoreColor(value) === 'var(--teal-400)' ? 'var(--gradient-brand)' : scoreColor(value) }} />
      </div>
    </div>
  );
}

// ─── Loading Spinner ──────────────────────────────────────────────────────────
function Spinner({ size = 16 }: { size?: number }) {
  return <RefreshCw size={size} className="spinning" style={{ color: 'var(--teal-400)' }} />;
}

// ─── Toast Notification ───────────────────────────────────────────────────────
function Toast({ msg, type, onClose }: { msg: string; type: 'success' | 'error'; onClose: () => void }) {
  useEffect(() => { const t = setTimeout(onClose, 4000); return () => clearTimeout(t); }, [onClose]);
  return (
    <div style={{
      position: 'fixed', bottom: 24, right: 24, zIndex: 999,
      display: 'flex', alignItems: 'center', gap: 10,
      background: type === 'success' ? 'rgba(20,184,166,0.15)' : 'rgba(248,113,113,0.15)',
      border: `1px solid ${type === 'success' ? 'rgba(20,184,166,0.3)' : 'rgba(248,113,113,0.3)'}`,
      borderRadius: 12, padding: '12px 18px', maxWidth: 380,
      backdropFilter: 'blur(16px)', boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
      animation: 'fadeIn 0.3s ease both'
    }}>
      {type === 'success' ? <CheckCircle2 size={16} color="var(--teal-400)" /> : <AlertCircle size={16} color="var(--red-400)" />}
      <span style={{ fontSize: 13, color: 'var(--text-primary)', flex: 1 }}>{msg}</span>
      <button onClick={onClose} className="btn-ghost" style={{ padding: '2px 4px' }}><X size={14} /></button>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
type View = 'knowledge' | 'chunking' | 'pipeline' | 'playground' | 'evaluation' | 'monitor' | 'settings';

const NAV_ITEMS: { id: View; label: string; icon: React.ElementType; section?: string }[] = [
  { id: 'knowledge',   label: 'Knowledge Sources',  icon: BookOpen,     section: 'Build' },
  { id: 'chunking',    label: 'Chunking Studio',     icon: Sliders,      section: 'Build' },
  { id: 'pipeline',    label: 'Pipeline Builder',    icon: GitBranch,    section: 'Build' },
  { id: 'playground',  label: 'Chat Playground',     icon: Terminal,     section: 'Explore' },
  { id: 'evaluation',  label: 'Evaluation Studio',   icon: FlaskConical, section: 'Evaluate' },
  { id: 'monitor',     label: 'Monitoring',          icon: BarChart2,    section: 'Observe' },
  { id: 'settings',    label: 'Settings',            icon: Settings,     section: 'Config' },
];

export default function DashboardApp() {
  const [view, setView] = useState<View>('knowledge');
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);
  const [config, setConfig] = useState<any>({ demo_mode: true });

  const showToast = useCallback((msg: string, type: 'success' | 'error' = 'success') => {
    setToast({ msg, type });
  }, []);

  useEffect(() => {
    apiFetch('/config').then(setConfig).catch(() => {});
  }, []);

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <nav className="sidebar">
        {/* Logo */}
        <div className="sidebar-logo">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ background: 'var(--gradient-brand)', borderRadius: 10, padding: 7, display: 'flex' }}>
              <Layers size={18} color="white" />
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 800, background: 'var(--gradient-brand)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                RAG Studio
              </div>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
                v2.0 · {config.demo_mode ? 'Demo Mode' : 'Live Mode'}
              </div>
            </div>
          </div>
        </div>

        {/* Nav Items */}
        <div className="sidebar-nav">
          {NAV_ITEMS.map((item, i) => {
            const prev = NAV_ITEMS[i - 1];
            const showSection = !prev || prev.section !== item.section;
            return (
              <React.Fragment key={item.id}>
                {showSection && <div className="nav-section-label">{item.section}</div>}
                <div
                  className={`nav-item ${view === item.id ? 'active' : ''}`}
                  onClick={() => setView(item.id)}
                >
                  <item.icon className="nav-icon" />
                  {item.label}
                </div>
              </React.Fragment>
            );
          })}
        </div>

        {/* Bottom status */}
        <div style={{ padding: '12px 14px', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 10, border: '1px solid var(--border)' }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--teal-400)', boxShadow: '0 0 8px var(--teal-400)' }} />
            <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 500 }}>Backend Online</span>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="main-content">
        {/* Top Bar */}
        <div className="topbar">
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
              {NAV_ITEMS.find(n => n.id === view)?.label}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }}>
              RAG Studio · Default Workspace
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {config.demo_mode && (
              <span className="badge badge-amber">Demo Mode</span>
            )}
            <span className="badge badge-teal">
              <Zap size={9} /> Live
            </span>
          </div>
        </div>

        {/* Page Content */}
        <div className="page-content fade-in" key={view}>
          {view === 'knowledge'   && <KnowledgeView showToast={showToast} />}
          {view === 'chunking'    && <ChunkingView showToast={showToast} />}
          {view === 'pipeline'    && <PipelineView showToast={showToast} />}
          {view === 'playground'  && <PlaygroundView showToast={showToast} />}
          {view === 'evaluation'  && <EvaluationView showToast={showToast} />}
          {view === 'monitor'     && <MonitorView showToast={showToast} />}
          {view === 'settings'    && <SettingsView config={config} showToast={showToast} />}
        </div>
      </div>

      {toast && <Toast msg={toast.msg} type={toast.type} onClose={() => setToast(null)} />}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// VIEW 1 — KNOWLEDGE SOURCES
// ══════════════════════════════════════════════════════════════════════════════
function KnowledgeView({ showToast }: { showToast: (m: string, t?: any) => void }) {
  const [docs, setDocs] = useState<any[]>([]);
  const [collections, setCollections] = useState<any[]>([]);
  const [selectedColl, setSelectedColl] = useState<number>(1);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [chunkStrategy, setChunkStrategy] = useState('recursive');
  const [chunkSize, setChunkSize] = useState(400);
  const [chunkOverlap, setChunkOverlap] = useState(50);

  const loadDocs = () => {
    apiFetch(`/documents?collection_id=${selectedColl}`).then(setDocs).catch(() => {});
  };
  const loadCollections = () => {
    apiFetch('/collections').then(cs => { setCollections(cs); if (cs.length > 0) setSelectedColl(cs[0].id); }).catch(() => {});
  };
  useEffect(() => { loadCollections(); }, []);
  useEffect(() => { loadDocs(); }, [selectedColl]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadProgress(['Parsing document...', 'Chunking text...', 'Generating embeddings...', 'Indexing vectors...']);
    const fd = new FormData();
    fd.append('collection_id', String(selectedColl));
    fd.append('chunk_strategy', chunkStrategy);
    fd.append('chunk_size', String(chunkSize));
    fd.append('chunk_overlap', String(chunkOverlap));
    fd.append('file', file);
    try {
      const result = await apiFetch('/documents/upload', { method: 'POST', body: fd });
      showToast(`✓ Uploaded ${result.document_name} — ${result.chunk_count} chunks indexed`);
      loadDocs();
    } catch (e: any) {
      showToast(e.message, 'error');
    } finally {
      setUploading(false);
      setUploadProgress([]);
    }
  };

  const handleDelete = async (docId: number) => {
    try {
      await apiFetch(`/documents/${docId}`, { method: 'DELETE' });
      showToast('Document deleted');
      loadDocs();
    } catch (e: any) {
      showToast(e.message, 'error');
    }
  };

  const onFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  const fileTypeIcon = (type: string) => {
    const icons: Record<string, string> = { pdf: '📄', txt: '📝', md: '📋', csv: '📊', json: '🗂', html: '🌐', docx: '📃' };
    return icons[type?.toLowerCase()] || '📁';
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: 20 }}>
      {/* Left: Upload */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Collection selector */}
        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Active Collection</div>
          <select className="input" value={selectedColl} onChange={e => setSelectedColl(Number(e.target.value))}>
            {collections.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </div>

        {/* Upload zone */}
        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--teal-400)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Upload size={14} /> Upload Documents
          </div>

          <div
            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onFileDrop}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={28} style={{ color: 'var(--teal-500)', margin: '0 auto 10px' }} />
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Drop file or click to browse</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>PDF · TXT · MD · HTML · CSV · JSON · DOCX</div>
            <input ref={fileRef} type="file" hidden onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])} accept=".pdf,.txt,.md,.html,.csv,.json,.docx,.markdown" />
          </div>

          {/* Ingestion settings */}
          <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div>
              <label className="field-label">Chunk Strategy</label>
              <select className="input" value={chunkStrategy} onChange={e => setChunkStrategy(e.target.value)}>
                <option value="recursive">Recursive (Recommended)</option>
                <option value="fixed">Fixed Character</option>
                <option value="semantic">Semantic (Sentence)</option>
                <option value="markdown">Markdown Headers</option>
                <option value="token">Token-Based</option>
              </select>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div>
                <label className="field-label">Chunk Size</label>
                <input type="number" className="input" value={chunkSize} onChange={e => setChunkSize(Number(e.target.value))} min={100} max={2000} step={50} />
              </div>
              <div>
                <label className="field-label">Overlap</label>
                <input type="number" className="input" value={chunkOverlap} onChange={e => setChunkOverlap(Number(e.target.value))} min={0} max={500} step={10} />
              </div>
            </div>
          </div>

          {/* Upload progress */}
          {uploading && (
            <div style={{ marginTop: 14, padding: 12, background: 'rgba(20,184,166,0.06)', border: '1px solid rgba(20,184,166,0.2)', borderRadius: 10 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--teal-400)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Processing Pipeline</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {uploadProgress.map((step, i) => (
                  <React.Fragment key={i}>
                    <span className="pipeline-step">
                      {i === uploadProgress.length - 1 ? <Spinner size={10} /> : <CheckCircle2 size={10} />}
                      {step}
                    </span>
                    {i < uploadProgress.length - 1 && <span className="pipeline-arrow">→</span>}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right: Document List */}
      <div className="panel" style={{ padding: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>Indexed Documents</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{docs.length} documents in collection</div>
          </div>
          <button className="btn-ghost" onClick={loadDocs}><RefreshCw size={13} /> Refresh</button>
        </div>

        {docs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-muted)' }}>
            <BookOpen size={36} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
            <div style={{ fontSize: 13 }}>No documents yet. Upload one to get started.</div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {docs.map((doc: any) => (
              <div key={doc.id} className="panel panel-hover" style={{ padding: 14, borderRadius: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', flex: 1 }}>
                    <div style={{ fontSize: 24, lineHeight: 1 }}>{fileTypeIcon(doc.file_type)}</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.name}</div>
                      <div style={{ display: 'flex', gap: 12, marginTop: 4, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{(doc.size_bytes / 1024).toFixed(1)} KB</span>
                        <span style={{ fontSize: 11, color: 'var(--teal-400)', fontWeight: 600 }}>{doc.chunk_count} chunks</span>
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{doc.word_count?.toLocaleString()} words</span>
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{doc.language?.toUpperCase()}</span>
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                    <span className={`badge ${doc.status === 'Completed' ? 'badge-teal' : doc.status === 'Processing' ? 'badge-amber' : 'badge-red'}`}>
                      {doc.status}
                    </span>
                    <button className="btn-ghost" onClick={() => handleDelete(doc.id)} style={{ padding: '4px 6px', color: 'var(--red-400)' }}>
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// VIEW 2 — CHUNKING STUDIO
// ══════════════════════════════════════════════════════════════════════════════
function ChunkingView({ showToast }: { showToast: (m: string, t?: any) => void }) {
  const [strategy, setStrategy] = useState('recursive');
  const [size, setSize] = useState(400);
  const [overlap, setOverlap] = useState(50);
  const [text, setText] = useState(`Retrieval-Augmented Generation (RAG) is an AI framework that combines large language models with external knowledge retrieval to produce more accurate, grounded responses.

RAG consists of two main components: a retriever and a generator. The retriever searches an external knowledge base for relevant information, while the generator (an LLM) uses this retrieved context to produce a well-grounded answer.

Key Benefits of RAG:
1. Reduces hallucinations by anchoring responses in retrieved facts
2. Enables use of private, proprietary, or recent data not in LLM training
3. Makes the system's knowledge sources transparent and auditable
4. Allows continuous knowledge updates without retraining the LLM

Vector Databases in RAG: Documents are split into chunks, converted to dense embedding vectors, and stored in a vector database like Qdrant, Pinecone, or pgvector. When a user asks a question, the query is also embedded and the vector database returns the most semantically similar chunks.`);
  const [chunks, setChunks] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [embedPreview, setEmbedPreview] = useState<any>(null);
  const [embedProvider, setEmbedProvider] = useState('google');
  const [providers, setProviders] = useState<any>({});

  useEffect(() => { apiFetch('/embeddings/providers').then(setProviders).catch(() => {}); }, []);

  const runPreview = async () => {
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append('text', text);
      fd.append('strategy', strategy);
      fd.append('size', String(size));
      fd.append('overlap', String(overlap));
      const data = await apiFetch('/chunker/preview', { method: 'POST', body: fd });
      setChunks(data.chunks);
    } catch (e: any) {
      showToast(e.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const runEmbedPreview = async () => {
    try {
      const sample = chunks[0]?.text_content || text.slice(0, 200);
      const fd = new FormData();
      fd.append('text', sample);
      fd.append('provider', embedProvider);
      const data = await apiFetch('/embeddings/generate', { method: 'POST', body: fd });
      setEmbedPreview(data);
    } catch (e: any) {
      showToast(e.message, 'error');
    }
  };

  useEffect(() => { runPreview(); }, [strategy, size, overlap]);

  const providerInfo = providers[embedProvider] || {};
  const strategies = [
    { id: 'recursive', label: 'Recursive', desc: 'Smart split at paragraph → sentence → word' },
    { id: 'fixed', label: 'Fixed Char', desc: 'Exact character count with overlap' },
    { id: 'semantic', label: 'Semantic', desc: 'Preserves sentence boundaries' },
    { id: 'markdown', label: 'Markdown', desc: 'Splits at ## headers' },
    { id: 'sentence', label: 'Sentence', desc: 'Sentence-level with overlap' },
    { id: 'token', label: 'Token', desc: 'Approximate token count (4 chars/token)' },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 20 }}>
      {/* Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--violet-400)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Sliders size={14} /> Strategy
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {strategies.map(s => (
              <button key={s.id} onClick={() => setStrategy(s.id)} style={{
                textAlign: 'left', padding: '9px 12px', borderRadius: 9,
                background: strategy === s.id ? 'rgba(139,92,246,0.1)' : 'transparent',
                border: `1px solid ${strategy === s.id ? 'rgba(139,92,246,0.3)' : 'transparent'}`,
                cursor: 'pointer', transition: 'all 0.15s'
              }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: strategy === s.id ? 'var(--violet-400)' : 'var(--text-secondary)' }}>{s.label}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{s.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--teal-400)', marginBottom: 14 }}>Parameters</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span className="field-label" style={{ marginBottom: 0 }}>Chunk Size (chars)</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--teal-400)', fontFamily: 'JetBrains Mono, monospace' }}>{size}</span>
              </div>
              <input type="range" min={100} max={2000} step={50} value={size} onChange={e => setSize(Number(e.target.value))} />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span className="field-label" style={{ marginBottom: 0 }}>Overlap (chars)</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--violet-400)', fontFamily: 'JetBrains Mono, monospace' }}>{overlap}</span>
              </div>
              <input type="range" min={0} max={300} step={10} value={overlap} onChange={e => setOverlap(Number(e.target.value))} />
            </div>
          </div>
        </div>

        {/* Embedding Preview */}
        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--teal-400)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Cpu size={13} /> Embedding Preview
          </div>
          <select className="input" value={embedProvider} onChange={e => setEmbedProvider(e.target.value)} style={{ marginBottom: 10 }}>
            {Object.entries(providers).map(([k, v]: any) => (
              <option key={k} value={k}>{k.toUpperCase()} — {v.model}</option>
            ))}
          </select>
          {providerInfo && (
            <div style={{ fontSize: 11, display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 10 }}>
              {[
                ['Dimensions', providerInfo.dimensions],
                ['MTEB Score', providerInfo.mteb_score],
                ['Context', `${providerInfo.context_window} tokens`],
                ['Cost/1k', `$${providerInfo.cost_per_1k_tokens}`],
              ].map(([label, val]) => (
                <div key={label as string} style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{label}</span>
                  <span style={{ fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-secondary)' }}>{String(val)}</span>
                </div>
              ))}
            </div>
          )}
          <button className="btn-secondary" style={{ width: '100%' }} onClick={runEmbedPreview}>
            <Zap size={12} /> Generate Embedding
          </button>
          {embedPreview && (
            <div style={{ marginTop: 10, padding: 10, background: 'var(--bg-base)', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6, fontWeight: 700 }}>VECTOR SAMPLE (first 10 dims)</div>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: 'var(--teal-400)', lineHeight: 1.6, wordBreak: 'break-all' }}>
                [{embedPreview.sample_values?.join(', ')}...]
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
                dims: {embedPreview.dimensions} · norm: {embedPreview.norm}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right: Text Input + Chunk Preview */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Input Text</span>
            <button className="btn-secondary" onClick={runPreview} disabled={loading}>
              {loading ? <Spinner size={12} /> : <Play size={12} />} Preview
            </button>
          </div>
          <textarea
            className="input"
            style={{ minHeight: 160, fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Paste any document text here to preview chunking..."
          />
        </div>

        <div className="panel" style={{ padding: 18, flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
              Generated Chunks
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span className="badge badge-teal">{chunks.length} chunks</span>
              {chunks.length > 0 && (
                <span className="badge badge-gray">{chunks.reduce((a, c) => a + c.token_count, 0)} tokens</span>
              )}
            </div>
          </div>
          <div style={{ maxHeight: 440, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {chunks.map((chunk: any, i: number) => (
              <div key={i} className="chunk-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Chunk #{i + 1}
                  </span>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <span style={{ fontSize: 10, color: 'var(--violet-400)', fontFamily: 'monospace', fontWeight: 600 }}>{chunk.token_count} tokens</span>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace' }}>{chunk.char_count} chars</span>
                  </div>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, fontStyle: 'italic' }}>
                  "{chunk.text_content}"
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// VIEW 3 — PIPELINE BUILDER
// ══════════════════════════════════════════════════════════════════════════════
function PipelineView({ showToast }: { showToast: (m: string, t?: any) => void }) {
  const [pipelines, setPipelines] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [providers, setProviders] = useState<any>({});
  const [llmProviders, setLlmProviders] = useState<any>({});

  useEffect(() => {
    apiFetch('/pipelines').then(ps => { setPipelines(ps); if (ps.length > 0) setSelected(ps[0]); }).catch(() => {});
    apiFetch('/embeddings/providers').then(setProviders).catch(() => {});
    apiFetch('/analytics/llm-providers').then(setLlmProviders).catch(() => {});
  }, []);

  const parseConfig = (cfg: string) => { try { return JSON.parse(cfg); } catch { return {}; } };

  if (!selected) return (
    <div style={{ textAlign: 'center', padding: '80px 0', color: 'var(--text-muted)' }}>
      <GitBranch size={40} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
      <div>No pipelines found</div>
    </div>
  );

  const chunk = parseConfig(selected.chunking_config);
  const embed = parseConfig(selected.embedding_config);
  const retriever = parseConfig(selected.retriever_config);
  const rerank = parseConfig(selected.rerank_config);
  const llm = parseConfig(selected.llm_config);

  const pipelineSteps = [
    { icon: '📤', label: 'Knowledge Source', detail: 'Documents & Collections', color: 'var(--teal-400)' },
    { icon: '✂️', label: 'Chunking', detail: `${chunk.strategy || 'recursive'} · ${chunk.size || 400} chars`, color: 'var(--violet-400)' },
    { icon: '🔢', label: 'Embedding', detail: `${embed.provider || 'google'} · ${embed.dimensions || 768}d`, color: 'var(--teal-400)' },
    { icon: '🗄️', label: 'Vector Store', detail: 'Qdrant (in-memory)', color: 'var(--violet-400)' },
    { icon: '🔍', label: 'Hybrid Retrieval', detail: `Dense ${retriever.dense_weight || 0.7} · Sparse ${retriever.sparse_weight || 0.3}`, color: 'var(--teal-400)' },
    { icon: '🏆', label: 'Re-ranking', detail: rerank.enabled ? `${rerank.model} · top-${rerank.top_n}` : 'Disabled', color: 'var(--violet-400)' },
    { icon: '🤖', label: 'LLM Generation', detail: llm.model || 'gemini/gemini-2.0-flash', color: 'var(--teal-400)' },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 20 }}>
      {/* Pipeline list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Pipelines</div>
        {pipelines.map(p => (
          <div key={p.id} onClick={() => setSelected(p)} style={{
            padding: '10px 12px', borderRadius: 10, cursor: 'pointer',
            background: selected?.id === p.id ? 'rgba(20,184,166,0.08)' : 'var(--bg-surface)',
            border: `1px solid ${selected?.id === p.id ? 'rgba(20,184,166,0.2)' : 'var(--border)'}`,
            transition: 'all 0.15s'
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: selected?.id === p.id ? 'var(--teal-400)' : 'var(--text-primary)' }}>{p.name}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{p.description || 'No description'}</div>
          </div>
        ))}
      </div>

      {/* Pipeline detail */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Visual Pipeline Flow */}
        <div className="panel" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Pipeline Architecture</div>
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 4 }}>
            {pipelineSteps.map((step, i) => (
              <React.Fragment key={i}>
                <div style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
                  padding: '10px 14px', borderRadius: 10,
                  background: `${step.color}0d`, border: `1px solid ${step.color}33`
                }}>
                  <span style={{ fontSize: 18 }}>{step.icon}</span>
                  <span style={{ fontSize: 10, fontWeight: 700, color: step.color, textAlign: 'center' }}>{step.label}</span>
                  <span style={{ fontSize: 9, color: 'var(--text-muted)', textAlign: 'center', maxWidth: 80 }}>{step.detail}</span>
                </div>
                {i < pipelineSteps.length - 1 && (
                  <ChevronRight size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Config Detail */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
          {[
            { title: 'Chunking Config', color: 'var(--violet-400)', items: [
              ['Strategy', chunk.strategy || 'recursive'],
              ['Size', `${chunk.size || 400} chars`],
              ['Overlap', `${chunk.overlap || 50} chars`],
            ]},
            { title: 'Embedding Config', color: 'var(--teal-400)', items: [
              ['Provider', embed.provider || 'google'],
              ['Model', embed.model || 'text-embedding-004'],
              ['Dimensions', embed.dimensions || 768],
            ]},
            { title: 'Retriever Config', color: 'var(--violet-400)', items: [
              ['Top K', retriever.top_k || 5],
              ['Dense Weight', retriever.dense_weight || 0.7],
              ['Sparse Weight', retriever.sparse_weight || 0.3],
            ]},
            { title: 'LLM Config', color: 'var(--teal-400)', items: [
              ['Provider', llm.provider || 'google'],
              ['Model', (llm.model || 'gemini/gemini-2.0-flash').split('/').pop()],
              ['Temperature', llm.temperature || 0.2],
            ]},
          ].map(card => (
            <div key={card.title} className="panel" style={{ padding: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: card.color, marginBottom: 12 }}>{card.title}</div>
              {card.items.map(([k, v]) => (
                <div key={k as string} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{k}</span>
                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'JetBrains Mono, monospace' }}>{String(v)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* System Prompt */}
        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--teal-400)', marginBottom: 10 }}>System Prompt</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, fontStyle: 'italic', padding: 12, background: 'var(--bg-base)', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
            "{selected.system_prompt}"
          </div>
        </div>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// VIEW 4 — CHAT PLAYGROUND
// ══════════════════════════════════════════════════════════════════════════════
function PlaygroundView({ showToast }: { showToast: (m: string, t?: any) => void }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<any[]>([
    { role: 'assistant', text: "👋 Welcome to the RAG Playground! Ask anything about your indexed documents. I'll show you exactly how I retrieved and reasoned over the context.", trace: null }
  ]);
  const [loading, setLoading] = useState(false);
  const [expandedTrace, setExpandedTrace] = useState<number | null>(null);
  const [expandedChunks, setExpandedChunks] = useState<number | null>(null);
  const [denseWeight, setDenseWeight] = useState(0.7);
  const [sparseWeight, setSparseWeight] = useState(0.3);
  const [searchResults, setSearchResults] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const pipelineId = 1;
  const collectionId = 1;

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append('query', userMsg);
      fd.append('pipeline_id', String(pipelineId));
      fd.append('collection_id', String(collectionId));
      const data = await apiFetch('/chat/playground', { method: 'POST', body: fd });
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer,
        chunks: data.retrieved_chunks,
        trace: data.explainability_trace,
        quality: data.generation_quality,
        latency: data.latency_ms,
        cost: data.cost_usd,
        tokens: data.total_tokens,
        demo: data.demo_mode,
      }]);
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', text: `❌ Error: ${e.message}`, trace: null }]);
      showToast(e.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const runSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const fd = new FormData();
      fd.append('query', searchQuery);
      fd.append('collection_id', String(collectionId));
      fd.append('dense_weight', String(denseWeight));
      fd.append('sparse_weight', String(sparseWeight));
      const data = await apiFetch('/search/compare', { method: 'POST', body: fd });
      setSearchResults(data);
    } catch (e: any) {
      showToast(e.message, 'error');
    } finally {
      setSearching(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 20, height: 'calc(100vh - 120px)' }}>
      {/* Left: Chat */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0, background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-xl)', overflow: 'hidden' }}>
        {/* Chat header */}
        <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--teal-400)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Terminal size={14} /> Chat Playground
          </div>
          <span className="badge badge-violet">Pipeline #1</span>
        </div>

        {/* Messages */}
        <div ref={chatRef} style={{ flex: 1, overflowY: 'auto', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {messages.map((msg: any, i: number) => (
            <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                {msg.role === 'user'
                  ? <div className="chat-bubble-user">{msg.text}</div>
                  : <div className="chat-bubble-assistant">{msg.text}</div>
                }
              </div>

              {/* Metadata row for assistant messages */}
              {msg.role === 'assistant' && msg.latency && (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
                    <Clock size={9} /> {msg.latency}ms
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
                    <DollarSign size={9} /> ${msg.cost?.toFixed(5)}
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
                    <Hash size={9} /> {msg.tokens} tokens
                  </span>
                  {msg.demo && <span className="badge badge-gray" style={{ fontSize: 9 }}>Demo</span>}
                </div>
              )}

              {/* Quality scores */}
              {msg.role === 'assistant' && msg.quality && (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {[
                    ['Faith', msg.quality.faithfulness],
                    ['Ground', msg.quality.groundedness],
                    ['Relevancy', msg.quality.answer_relevancy],
                  ].map(([k, v]: any) => (
                    <span key={k} className={`badge ${scoreBadge(v)}`}>{k}: {pct(v)}</span>
                  ))}
                  {msg.quality.hallucination_rate < 0.1 && (
                    <span className="badge badge-teal">✓ Low hallucination</span>
                  )}
                </div>
              )}

              {/* Retrieved chunks toggle */}
              {msg.chunks?.length > 0 && (
                <div>
                  <button onClick={() => setExpandedChunks(expandedChunks === i ? null : i)}
                    style={{ fontSize: 11, color: 'var(--violet-400)', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                    {expandedChunks === i ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    {msg.chunks.length} Retrieved Chunks
                  </button>
                  {expandedChunks === i && (
                    <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {msg.chunks.map((c: any, ci: number) => (
                        <div key={ci} className="result-card highlighted">
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                            <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)' }}>RANK #{ci + 1} · {c.source || 'unknown'}</span>
                            <span style={{ fontSize: 10, color: 'var(--teal-400)', fontWeight: 700, fontFamily: 'monospace' }}>
                              {(c.rerank_score || c.rrf_score || c.score || 0).toFixed(3)}
                            </span>
                          </div>
                          <p style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5, fontStyle: 'italic' }}>"{c.text?.slice(0, 160)}..."</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Explainability trace toggle */}
              {msg.trace && (
                <div>
                  <button onClick={() => setExpandedTrace(expandedTrace === i ? null : i)}
                    style={{ fontSize: 11, color: 'var(--teal-400)', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                    {expandedTrace === i ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    Explain Retrieval Steps
                  </button>
                  {expandedTrace === i && (
                    <div className="trace-panel" style={{ marginTop: 8 }}>
                      {msg.trace.steps?.map((step: any, si: number) => (
                        <div key={si} className="trace-step">
                          <div className="trace-dot" />
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 700, color: 'var(--teal-400)', textTransform: 'uppercase', fontSize: 10, letterSpacing: '0.05em' }}>
                              {step.step.replace(/_/g, ' ')}
                            </div>
                            <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>
                              {Object.entries(step).filter(([k]) => k !== 'step').map(([k, v]) => (
                                <span key={k} style={{ marginRight: 12 }}>
                                  <span style={{ color: 'var(--text-muted)' }}>{k}: </span>
                                  <span style={{ color: 'var(--text-secondary)', fontFamily: 'monospace', fontWeight: 600 }}>{String(v)}</span>
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div className="chat-bubble-assistant" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Spinner size={14} /> Retrieving context and generating response...
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <form onSubmit={sendMessage} style={{ padding: '12px 16px', borderTop: '1px solid var(--border-subtle)', display: 'flex', gap: 8 }}>
          <input
            className="input"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask anything about your indexed documents..."
            disabled={loading}
          />
          <button type="submit" className="btn-primary" disabled={loading || !input.trim()} style={{ flexShrink: 0 }}>
            {loading ? <Spinner size={14} /> : <Send size={14} />}
          </button>
        </form>
      </div>

      {/* Right: Hybrid Search Panel */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, overflowY: 'auto' }}>
        <div className="panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--teal-400)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Search size={13} /> Hybrid Search Tuner
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 14 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span className="field-label" style={{ marginBottom: 0 }}>Dense (Semantic)</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--teal-400)', fontFamily: 'monospace' }}>{denseWeight.toFixed(2)}</span>
              </div>
              <input type="range" min={0} max={1} step={0.05} value={denseWeight} onChange={e => { const v = Number(e.target.value); setDenseWeight(v); setSparseWeight(+(1-v).toFixed(2)); }} />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span className="field-label" style={{ marginBottom: 0 }}>Sparse (BM25)</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--violet-400)', fontFamily: 'monospace' }}>{sparseWeight.toFixed(2)}</span>
              </div>
              <input type="range" min={0} max={1} step={0.05} value={sparseWeight} onChange={e => { const v = Number(e.target.value); setSparseWeight(v); setDenseWeight(+(1-v).toFixed(2)); }} />
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            <input className="input" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} placeholder="Test query..." style={{ flex: 1 }} onKeyDown={e => e.key === 'Enter' && runSearch()} />
            <button className="btn-secondary" onClick={runSearch} disabled={searching} style={{ flexShrink: 0 }}>
              {searching ? <Spinner size={12} /> : <Search size={12} />}
            </button>
          </div>

          {searchResults && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                { label: 'Dense Results', color: 'var(--teal-400)', data: searchResults.dense },
                { label: 'Sparse (BM25)', color: 'var(--violet-400)', data: searchResults.sparse },
                { label: '⚡ Fused (RRF)', color: 'var(--amber-400)', data: searchResults.fused },
              ].map(({ label, color, data }) => (
                <div key={label} style={{ borderRadius: 10, border: `1px solid ${color}33`, overflow: 'hidden' }}>
                  <div style={{ padding: '6px 10px', background: `${color}11`, fontSize: 10, fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
                  <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {(data || []).slice(0, 3).map((r: any, ri: number) => (
                      <div key={ri} style={{ padding: '6px 8px', background: 'var(--bg-base)', borderRadius: 7, border: '1px solid var(--border-subtle)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 700 }}>#{ri + 1}</span>
                          <span style={{ fontSize: 9, color, fontFamily: 'monospace', fontWeight: 700 }}>
                            {(r.rrf_score || r.score || 0).toFixed(4)}
                          </span>
                        </div>
                        <p style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.4 }}>"{(r.text || '').slice(0, 80)}..."</p>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// VIEW 5 — EVALUATION STUDIO
// ══════════════════════════════════════════════════════════════════════════════
function EvaluationView({ showToast }: { showToast: (m: string, t?: any) => void }) {
  const [runs, setRuns] = useState<any[]>([]);
  const [running, setRunning] = useState(false);
  const [selected, setSelected] = useState<any>(null);

  const loadRuns = () => {
    apiFetch('/evaluations').then(r => { setRuns(r); if (r.length > 0 && !selected) setSelected(r[0]); }).catch(() => {});
  };
  useEffect(loadRuns, []);

  const triggerRun = async () => {
    setRunning(true);
    try {
      const fd = new FormData();
      fd.append('pipeline_id', '1');
      fd.append('name', `Experiment ${runs.length + 1} — ${new Date().toLocaleTimeString()}`);
      const run = await apiFetch('/evaluations/run', { method: 'POST', body: fd });
      showToast(`✓ Evaluation "${run.name}" completed`);
      loadRuns();
      setSelected(run);
    } catch (e: any) {
      showToast(e.message, 'error');
    } finally {
      setRunning(false);
    }
  };

  const radarData = selected ? [
    { metric: 'Faithfulness', value: Math.round(selected.faithfulness * 100) },
    { metric: 'Groundedness', value: Math.round(selected.groundedness * 100) },
    { metric: 'Relevancy', value: Math.round(selected.answer_relevancy * 100) },
    { metric: 'Recall@K', value: Math.round(selected.recall_at_k * 100) },
    { metric: 'Precision', value: Math.round(selected.precision_at_k * 100) },
    { metric: 'MRR', value: Math.round(selected.mrr * 100) },
  ] : [];

  const RADAR_COLORS = ['#14b8a6', '#8b5cf6'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div className="page-title">Evaluation Studio</div>
          <div className="page-subtitle">Compare pipeline experiments side-by-side with RAG quality metrics</div>
        </div>
        <button className="btn-primary" onClick={triggerRun} disabled={running}>
          {running ? <><Spinner size={14} /> Running...</> : <><Play size={14} /> Run Evaluation</>}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 20 }}>
        {/* Left: Table */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="panel" style={{ overflow: 'hidden' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Experiment</th>
                  <th>Recall@K</th>
                  <th>Faithfulness</th>
                  <th>Groundedness</th>
                  <th>Halluc. Rate</th>
                  <th>Latency</th>
                  <th>Queries</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run: any) => (
                  <tr key={run.id} onClick={() => setSelected(run)} style={{ cursor: 'pointer', background: selected?.id === run.id ? 'rgba(20,184,166,0.06)' : undefined }}>
                    <td style={{ fontWeight: 600, maxWidth: 200 }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{run.name}</div>
                    </td>
                    <td><span className={`badge ${scoreBadge(run.recall_at_k)}`}>{pct(run.recall_at_k)}</span></td>
                    <td><span className={`badge ${scoreBadge(run.faithfulness)}`}>{pct(run.faithfulness)}</span></td>
                    <td><span className={`badge ${scoreBadge(run.groundedness)}`}>{pct(run.groundedness)}</span></td>
                    <td><span className={`badge ${run.hallucination_rate < 0.1 ? 'badge-teal' : 'badge-red'}`}>{pct(run.hallucination_rate)}</span></td>
                    <td style={{ fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{run.avg_latency_ms}ms</td>
                    <td style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>{run.queries_evaluated}</td>
                    <td><span className={`badge ${run.status === 'Completed' ? 'badge-teal' : run.status === 'Running' ? 'badge-amber' : 'badge-red'}`}>{run.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Score bars for selected */}
          {selected && (
            <div className="panel" style={{ padding: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>
                Metric Breakdown: <span style={{ color: 'var(--teal-400)' }}>{selected.name}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {[
                  ['Recall@K', selected.recall_at_k],
                  ['Precision@K', selected.precision_at_k],
                  ['MRR', selected.mrr],
                  ['nDCG', selected.ndcg],
                  ['Faithfulness', selected.faithfulness],
                  ['Groundedness', selected.groundedness],
                  ['Answer Relevancy', selected.answer_relevancy],
                  ['Citation Accuracy', selected.citation_accuracy],
                ].map(([label, val]: any) => (
                  <ScoreBar key={label} label={label} value={val || 0} />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Radar chart */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {selected && radarData.length > 0 && (
            <div className="panel" style={{ padding: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 14 }}>Quality Radar</div>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Radar dataKey="value" stroke="#14b8a6" fill="#14b8a6" fillOpacity={0.15} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}

          {selected && (
            <div className="panel" style={{ padding: 18 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12 }}>Cost & Performance</div>
              {[
                ['Avg Latency', `${selected.avg_latency_ms}ms`, 'var(--teal-400)'],
                ['Avg Cost/Query', `$${selected.avg_cost_usd?.toFixed(6)}`, 'var(--violet-400)'],
                ['Total Tokens', selected.total_tokens?.toLocaleString(), 'var(--text-secondary)'],
                ['Queries Evaluated', selected.queries_evaluated, 'var(--text-secondary)'],
                ['Hallucination Rate', pct(selected.hallucination_rate), selected.hallucination_rate < 0.1 ? 'var(--teal-400)' : 'var(--red-400)'],
              ].map(([k, v, c]) => (
                <div key={k as string} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{k}</span>
                  <span style={{ fontSize: 12, fontWeight: 700, color: c as string, fontFamily: 'monospace' }}>{String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// VIEW 6 — MONITORING DASHBOARD
// ══════════════════════════════════════════════════════════════════════════════
function MonitorView({ showToast }: { showToast: (m: string, t?: any) => void }) {
  const [analytics, setAnalytics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch('/analytics').then(d => { setAnalytics(d); setLoading(false); }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ textAlign: 'center', padding: 60 }}><Spinner size={28} /></div>;
  if (!analytics) return null;

  const queryData = (analytics.queries_per_day || []).map((v: number, i: number) => ({ day: `D${i + 1}`, queries: v }));
  const costData = [
    { name: 'Embedding', value: analytics.total_cost_usd * 0.12, color: '#14b8a6' },
    { name: 'LLM', value: analytics.total_cost_usd * 0.78, color: '#8b5cf6' },
    { name: 'Reranking', value: analytics.total_cost_usd * 0.10, color: '#fbbf24' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <div className="page-title">Monitoring Dashboard</div>
        <div className="page-subtitle">Real-time analytics, costs, and feedback from your RAG pipelines</div>
      </div>

      {/* KPI Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14 }}>
        {[
          { icon: Activity, label: 'Total Queries', value: analytics.total_queries?.toLocaleString(), delta: '▲ 14%', pos: true },
          { icon: Clock, label: 'Avg Latency', value: `${analytics.avg_latency_ms}ms`, delta: '▼ 12ms reduction', pos: true },
          { icon: DollarSign, label: 'Total Cost', value: `$${analytics.total_cost_usd?.toFixed(4)}`, delta: 'All providers', pos: null },
          { icon: ThumbsUp, label: 'Satisfaction', value: `${((analytics.positive_feedback_rate || 0.94) * 100).toFixed(1)}%`, delta: 'User feedback', pos: true },
        ].map(({ icon: Icon, label, value, delta, pos }) => (
          <div key={label} className="metric-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="metric-label">{label}</span>
              <Icon size={16} style={{ color: 'var(--teal-400)' }} />
            </div>
            <div className="metric-value">{value}</div>
            <div className={`metric-delta ${pos === true ? 'positive' : pos === false ? 'negative' : ''}`}>{delta}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 320px', gap: 16 }}>
        {/* Query chart */}
        <div className="panel" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Daily Queries</div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={queryData}>
              <defs>
                <linearGradient id="qGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11 }} />
              <Area type="monotone" dataKey="queries" stroke="#14b8a6" fill="url(#qGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Cost breakdown */}
        <div className="panel" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Cost Breakdown</div>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            <ResponsiveContainer width={140} height={140}>
              <PieChart>
                <Pie data={costData} innerRadius={40} outerRadius={65} dataKey="value" paddingAngle={3}>
                  {costData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {costData.map(d => (
                <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: d.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 11, color: 'var(--text-secondary)', flex: 1 }}>{d.name}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>${d.value.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Health stats */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[
            { label: 'Cache Hit Rate', value: analytics.cache_hit_rate || 0.24, icon: Zap, color: 'var(--teal-400)' },
            { label: 'Hallucination Score', value: analytics.avg_hallucination_score || 0.03, icon: ShieldAlert, color: analytics.avg_hallucination_score < 0.1 ? 'var(--teal-400)' : 'var(--red-400)' },
          ].map(({ label, value, icon: Icon, color }) => (
            <div key={label} className="panel" style={{ padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 600 }}>{label}</span>
                <Icon size={14} style={{ color }} />
              </div>
              <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: 'monospace', marginBottom: 8 }}>{pct(value)}</div>
              <div className="score-bar">
                <div className="score-bar-fill" style={{ width: `${value * 100}%`, background: color }} />
              </div>
            </div>
          ))}

          <div className="panel" style={{ padding: 16 }}>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 600, marginBottom: 10 }}>Recent Queries</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {(analytics.top_queries || []).map((q: string, i: number) => (
                <div key={i} style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                  <MessageSquare size={10} style={{ flexShrink: 0, marginTop: 1, color: 'var(--teal-400)' }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{q}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Feedback section */}
      <div className="panel" style={{ padding: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Human Feedback Loop</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
          {[
            { icon: ThumbsUp, label: '👍 Positive Feedback', value: `${(analytics.positive_feedback_rate * 100 || 94.8).toFixed(1)}%`, color: 'var(--teal-400)' },
            { icon: ShieldAlert, label: '⚠️ Hallucination Rate', value: pct(analytics.avg_hallucination_score || 0.03), color: analytics.avg_hallucination_score < 0.1 ? 'var(--teal-400)' : 'var(--amber-400)' },
            { icon: Activity, label: '📊 Total Queries Logged', value: analytics.total_queries?.toLocaleString() || '0', color: 'var(--violet-400)' },
          ].map(({ icon: Icon, label, value, color }) => (
            <div key={label} style={{ padding: 16, background: 'var(--bg-elevated)', borderRadius: 12, border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: 14 }}>
              <div style={{ padding: 10, background: `${color}15`, borderRadius: 10, border: `1px solid ${color}30` }}>
                <Icon size={18} style={{ color }} />
              </div>
              <div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
                <div style={{ fontSize: 20, fontWeight: 800, color, marginTop: 2 }}>{value}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// VIEW 7 — SETTINGS
// ══════════════════════════════════════════════════════════════════════════════
function SettingsView({ config, showToast }: { config: any; showToast: (m: string, t?: any) => void }) {
  const sections = [
    {
      title: 'API Configuration',
      icon: Key,
      items: [
        { label: 'Demo Mode', value: config.demo_mode ? '✓ Enabled (no API keys needed)' : '✗ Disabled (real API calls)', ok: true },
        { label: 'Google Gemini API Key', value: config.has_google_key ? '✓ Configured' : '✗ Not set (set GOOGLE_API_KEY in .env)', ok: config.has_google_key },
        { label: 'OpenAI API Key', value: config.has_openai_key ? '✓ Configured' : '✗ Not set (optional)', ok: config.has_openai_key },
        { label: 'Cohere API Key', value: config.has_cohere_key ? '✓ Configured' : '✗ Not set (optional — enables real reranking)', ok: config.has_cohere_key },
      ]
    },
    {
      title: 'Infrastructure',
      icon: Database,
      items: [
        { label: 'Database', value: 'SQLite (rag_studio.db) — zero setup', ok: true },
        { label: 'Vector Store', value: 'Qdrant in-memory — zero setup', ok: true },
        { label: 'Embedding Dimensions', value: String(config.embedding_dimensions || 768), ok: true },
        { label: 'Default LLM', value: config.default_llm_model || 'gemini/gemini-2.0-flash', ok: true },
      ]
    }
  ];

  return (
    <div style={{ maxWidth: 700 }}>
      <div style={{ marginBottom: 24 }}>
        <div className="page-title">Settings</div>
        <div className="page-subtitle">Configure API keys, providers, and system defaults</div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {sections.map(section => (
          <div key={section.title} className="panel" style={{ padding: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--teal-400)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
              <section.icon size={14} /> {section.title}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {section.items.map(item => (
                <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px', background: 'var(--bg-base)', borderRadius: 9, border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{item.label}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: item.ok ? 'var(--teal-400)' : 'var(--amber-400)' }}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* Setup guide */}
        <div className="panel" style={{ padding: 20, border: '1px solid rgba(20,184,166,0.2)' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--teal-400)', marginBottom: 12 }}>🚀 Quick Setup Guide</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            <p style={{ marginBottom: 8 }}>1. Copy <code style={{ fontFamily: 'monospace', color: 'var(--teal-400)', background: 'rgba(20,184,166,0.1)', padding: '1px 5px', borderRadius: 4 }}>backend/.env.example</code> to <code style={{ fontFamily: 'monospace', color: 'var(--teal-400)', background: 'rgba(20,184,166,0.1)', padding: '1px 5px', borderRadius: 4 }}>backend/.env</code></p>
            <p style={{ marginBottom: 8 }}>2. Get a free Google AI API key at <a href="https://aistudio.google.com" target="_blank" rel="noreferrer" style={{ color: 'var(--teal-400)' }}>aistudio.google.com</a></p>
            <p style={{ marginBottom: 8 }}>3. Set <code style={{ fontFamily: 'monospace', color: 'var(--violet-400)', background: 'rgba(139,92,246,0.1)', padding: '1px 5px', borderRadius: 4 }}>GOOGLE_API_KEY=your_key</code> and <code style={{ fontFamily: 'monospace', color: 'var(--violet-400)', background: 'rgba(139,92,246,0.1)', padding: '1px 5px', borderRadius: 4 }}>DEMO_MODE=false</code></p>
            <p>4. Restart the backend — real Gemini embeddings and LLM responses will be used!</p>
          </div>
        </div>
      </div>
    </div>
  );
}
