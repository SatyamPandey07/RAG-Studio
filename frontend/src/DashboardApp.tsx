import React, { useState, useMemo } from 'react';
import { 
  Database, 
  Layers, 
  Cpu, 
  Sliders, 
  Settings, 
  Plus, 
  Search, 
  Send, 
  PieChart, 
  Compass, 
  TrendingUp, 
  ThumbsUp, 
  ThumbsDown, 
  FileText, 
  Sparkles, 
  Check, 
  ChevronRight, 
  HelpCircle, 
  Terminal, 
  Activity, 
  ShieldAlert, 
  Key, 
  Play, 
  Network
} from 'lucide-react';

export default function DashboardApp() {
  const [activeTab, setActiveTab] = useState<'design' | 'evaluate' | 'monitor'>('design');
  const [workspace, setWorkspace] = useState('Default Workspace');
  const [project, setProject] = useState('Support Assistant RAG');
  
  // Module 5 State: Chunking Studio
  const [chunkStrategy, setChunkStrategy] = useState<'recursive' | 'fixed' | 'semantic'>('recursive');
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const sampleDocumentText = `Retrieval-Augmented Generation (RAG) is an AI framework for retrieving information from an external knowledge source to anchor Large Language Models (LLMs) on factual, domain-specific data. RAG addresses the core limitations of LLMs, such as hallucinations, training cut-off dates, and lack of access to private enterprise documents. By splitting documents into small chunks, converting them to dense embeddings, and storing them in vector databases, RAG retrieves the most relevant snippets relative to a user's question, feeding them as context.`;
  
  // Module 6 & 7 State: Embeddings & Vector DBs
  const [selectedEmbedder, setSelectedEmbedder] = useState('google');
  const [selectedVectorDb, setSelectedVectorDb] = useState('qdrant');
  
  // Module 8 State: Hybrid Search Weights
  const [denseWeight, setDenseWeight] = useState(0.7);
  const [sparseWeight, setSparseWeight] = useState(0.3);
  const [searchQuery, setSearchQuery] = useState('How does RAG solve hallucinations?');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<any>(null);

  // Module 11 & 12 State: Prompt & LLM
  const [systemPrompt, setSystemPrompt] = useState('You are an expert system. Answer the query using ONLY the retrieved chunks.');
  const [selectedLlm, setSelectedLlm] = useState('google-gemini-flash');
  
  // Module 13 & 14 State: Chat & Explainability
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState<any[]>([
    {
      role: 'agent',
      text: "Hello! Welcome to the RAG Studio playground. Ask me anything about your loaded documents, and I'll explain my retrieval steps.",
      trace: null
    }
  ]);
  const [showTraceForIndex, setShowTraceForIndex] = useState<number | null>(null);

  // Module 15 State: Evaluation experiments
  const [evalRuns, setEvalRuns] = useState<any[]>([
    { name: "Recursive + Google Embeddings", strategy: "Recursive", size: 500, overlap: 50, recall: 0.88, faithfulness: 0.94, cost: 0.012, latency: 310 },
    { name: "Fixed + OpenAI Embeddings", strategy: "Fixed", size: 256, overlap: 20, recall: 0.81, faithfulness: 0.88, cost: 0.024, latency: 450 }
  ]);
  
  // Documents State
  const [documents, setDocuments] = useState<any[]>([
    { name: "rag_architecture.pdf", size: "142 KB", chunks: 12, date: "2026-06-27" },
    { name: "vectordb_indexing.txt", size: "48 KB", chunks: 5, date: "2026-06-26" }
  ]);
  const [uploading, setUploading] = useState(false);
  const [uploadedName, setUploadedName] = useState('');

  // 1. Chunking computation preview
  const chunksPreview = useMemo(() => {
    let result: string[] = [];
    if (chunkStrategy === 'fixed') {
      let start = 0;
      while (start < sampleDocumentText.length) {
        result.push(sampleDocumentText.slice(start, start + chunkSize));
        start += chunkSize - chunkOverlap;
        if (chunkOverlap >= chunkSize) start += 1;
      }
    } else if (chunkStrategy === 'semantic') {
      result = sampleDocumentText.split(/(?<=[.!?])\s+/);
    } else { // recursive
      const words = sampleDocumentText.split(' ');
      let current = '';
      for (const word of words) {
        if ((current + ' ' + word).length <= chunkSize) {
          current += (current ? ' ' : '') + word;
        } else {
          result.push(current);
          current = word;
        }
      }
      if (current) result.push(current);
    }
    return result.map((content, idx) => ({
      index: idx,
      text_content: content,
      token_count: Math.max(1, Math.floor(content.length / 4))
    }));
  }, [chunkStrategy, chunkSize, chunkOverlap]);

  // 2. Hybrid search fusion calculation
  const runMockSearch = () => {
    setIsSearching(true);
    setTimeout(() => {
      const dense = [
        { id: "c1", score: 0.92, text: "RAG retrieves relevant snippets relative to a user's question, feeding them as context.", source: "rag_architecture.pdf" },
        { id: "c2", score: 0.81, "text": "Converting text chunks to dense embeddings and storing them in vector databases.", "source": "vectordb_indexing.txt" }
      ];
      const sparse = [
        { id: "c1", score: 0.78, text: "RAG retrieves relevant snippets relative to a user's question, feeding them as context.", source: "rag_architecture.pdf" },
        { id: "c3", score: 0.69, text: "RAG addresses core limitations of LLMs, such as hallucinations and Cut-Off dates.", source: "rag_architecture.pdf" }
      ];

      // Fusion algorithm (linear score weight combined with Reciprocal Rank Fusion ranks)
      const fused = [
        { id: "c1", score: (0.92 * denseWeight + 0.78 * sparseWeight).toFixed(3), text: "RAG retrieves relevant snippets relative to a user's question, feeding them as context.", source: "rag_architecture.pdf", rank: 1 },
        { id: "c2", score: (0.81 * denseWeight).toFixed(3), "text": "Converting text chunks to dense embeddings and storing them in vector databases.", "source": "vectordb_indexing.txt", rank: 2 },
        { id: "c3", score: (0.69 * sparseWeight).toFixed(3), text: "RAG addresses core limitations of LLMs, such as hallucinations and Cut-Off dates.", source: "rag_architecture.pdf", rank: 3 }
      ];

      setSearchResults({ dense, sparse, fused });
      setIsSearching(false);
    }, 600);
  };

  // 3. Playground Chat messaging & Explainability trace
  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = { role: 'user', text: chatInput };
    setChatHistory(prev => [...prev, userMsg]);
    const currentInput = chatInput;
    setChatInput('');

    setTimeout(() => {
      // Create detailed trace object
      const trace = {
        query: currentInput,
        retrieved: [
          { rank: 1, text: "RAG addresses the core limitations of LLMs, such as hallucinations...", source: "rag_architecture.pdf", similarity: 0.94 },
          { rank: 2, text: "By splitting documents into small chunks, converting them to dense embeddings...", source: "rag_architecture.pdf", similarity: 0.86 }
        ],
        denseWeight,
        sparseWeight,
        rerankModel: "Cohere-Rerank-v3",
        latencyMs: 145,
        tokenUsage: 1200,
        hallucinationIndex: 0.02
      };

      setChatHistory(prev => [...prev, {
        role: 'agent',
        text: `Based on your documents, RAG solves hallucinations by anchoring responses in retrieved facts (e.g. splitting files into chunks, creating dense embeddings, and querying vector databases) to provide grounding context for the model.`,
        trace: trace
      }]);
    }, 800);
  };

  // 4. Ingestion simulator
  const handleUpload = (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadedName.trim()) return;
    setUploading(true);
    setTimeout(() => {
      const newDoc = {
        name: uploadedName.endsWith('.pdf') || uploadedName.endsWith('.txt') ? uploadedName : `${uploadedName}.pdf`,
        size: "112 KB",
        chunks: Math.floor(Math.random() * 8) + 4,
        date: new Date().toISOString().split('T')[0]
      };
      setDocuments(prev => [newDoc, ...prev]);
      setUploadedName('');
      setUploading(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col min-h-screen text-slate-200">
      
      {/* Top Header Section */}
      <header className="glass-panel border-b border-white/5 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="bg-gradient-to-r from-teal-500 to-violet-500 p-2.5 rounded-xl shadow-elegant">
            <Layers className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-teal-400 via-violet-400 to-teal-400 bg-clip-text text-transparent">
              RAG Studio
            </h1>
            <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">
              Production-Grade RAG Engine
            </p>
          </div>
        </div>

        {/* Workspace and Project Selectors */}
        <div className="hidden md:flex items-center space-x-4">
          <div className="flex flex-col">
            <span className="text-[9px] uppercase font-bold text-slate-500">Workspace</span>
            <select 
              value={workspace} 
              onChange={(e) => setWorkspace(e.target.value)}
              className="bg-slate-900 border border-white/10 rounded px-2 py-1 text-sm outline-none text-teal-400 font-semibold"
            >
              <option>Default Workspace</option>
              <option>Enterprise RAG</option>
              <option>Developer Sandbox</option>
            </select>
          </div>

          <div className="flex flex-col">
            <span className="text-[9px] uppercase font-bold text-slate-500">Active Project</span>
            <select 
              value={project} 
              onChange={(e) => setProject(e.target.value)}
              className="bg-slate-900 border border-white/10 rounded px-2 py-1 text-sm outline-none text-violet-400 font-semibold"
            >
              <option>Support Assistant RAG</option>
              <option>Internal Wiki Q&A</option>
              <option>Sales Doc Analyzer</option>
            </select>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-2">
          <button 
            onClick={() => setActiveTab('design')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
              activeTab === 'design' 
                ? 'bg-gradient-to-r from-teal-500/20 to-violet-500/20 border border-teal-500/30 text-teal-400 shadow-glow' 
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            Design & Build
          </button>
          <button 
            onClick={() => setActiveTab('evaluate')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
              activeTab === 'evaluate' 
                ? 'bg-gradient-to-r from-teal-500/20 to-violet-500/20 border border-teal-500/30 text-teal-400 shadow-glow' 
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            Evaluate
          </button>
          <button 
            onClick={() => setActiveTab('monitor')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
              activeTab === 'monitor' 
                ? 'bg-gradient-to-r from-teal-500/20 to-violet-500/20 border border-teal-500/30 text-teal-400 shadow-glow' 
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            Monitor
          </button>
        </nav>
      </header>

      {/* Main Panel Content Area */}
      <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-6">
        
        {/* TAB 1: DESIGN & BUILD */}
        {activeTab === 'design' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left side: Upload & Ingestion Pipeline */}
            <div className="lg:col-span-4 space-y-6">
              <section className="glass-panel rounded-2xl p-5 shadow-elegant">
                <h3 className="text-base font-bold mb-4 flex items-center text-teal-400">
                  <FileText className="mr-2 h-5 w-5" /> Knowledge Sources
                </h3>
                
                {/* Upload Form */}
                <form onSubmit={handleUpload} className="mb-5">
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={uploadedName}
                      onChange={(e) => setUploadedName(e.target.value)}
                      placeholder="e.g. product_guide.pdf"
                      className="bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm flex-1 outline-none focus:border-teal-500 transition"
                    />
                    <button 
                      type="submit" 
                      disabled={uploading}
                      className="bg-gradient-to-r from-teal-500 to-violet-500 text-white px-4 py-2 rounded-lg text-sm font-bold hover:brightness-110 active:scale-95 transition disabled:opacity-50 flex items-center"
                    >
                      {uploading ? 'Scanning...' : 'Upload'}
                    </button>
                  </div>
                </form>

                {/* Upload progress visualization */}
                {uploading && (
                  <div className="bg-slate-900 border border-teal-500/20 rounded-xl p-3 mb-4 animate-pulse">
                    <span className="text-[10px] font-bold text-teal-400 uppercase tracking-widest block mb-2">Ingestion Pipeline</span>
                    <div className="flex items-center space-x-2 text-xs text-slate-300">
                      <span className="bg-teal-500/20 text-teal-300 px-1.5 py-0.5 rounded">OCR</span>
                      <ChevronRight className="h-3 w-3 text-slate-500" />
                      <span className="bg-teal-500/20 text-teal-300 px-1.5 py-0.5 rounded">Deduplication</span>
                      <ChevronRight className="h-3 w-3 text-slate-500" />
                      <span className="bg-violet-500/20 text-violet-300 px-1.5 py-0.5 rounded animate-bounce">Chunking</span>
                    </div>
                  </div>
                )}

                {/* List of Documents */}
                <div className="space-y-3">
                  {documents.map((doc, idx) => (
                    <div key={idx} className="bg-white/5 border border-white/5 rounded-xl p-3 flex justify-between items-center">
                      <div>
                        <div className="text-sm font-semibold">{doc.name}</div>
                        <div className="text-[10px] text-slate-400 mt-1">{doc.size} · {doc.chunks} chunks</div>
                      </div>
                      <span className="bg-teal-500/10 text-teal-400 border border-teal-500/20 text-[10px] font-bold px-2 py-0.5 rounded-full">
                        Ingested
                      </span>
                    </div>
                  ))}
                </div>
              </section>

              {/* Chunking Studio */}
              <section className="glass-panel rounded-2xl p-5 shadow-elegant">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-base font-bold flex items-center text-violet-400">
                    <Sliders className="mr-2 h-5 w-5" /> Chunking Studio
                  </h3>
                  <select 
                    value={chunkStrategy} 
                    onChange={(e: any) => setChunkStrategy(e.target.value)}
                    className="bg-slate-900 border border-white/10 rounded px-2 py-1 text-xs outline-none text-slate-300"
                  >
                    <option value="recursive">Recursive</option>
                    <option value="fixed">Fixed Character</option>
                    <option value="semantic">Semantic (Sentence)</option>
                  </select>
                </div>

                {/* Sliders */}
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">Chunk Size (chars)</span>
                      <span className="text-teal-400 font-mono font-bold">{chunkSize}</span>
                    </div>
                    <input 
                      type="range" min="100" max="1500" step="50"
                      value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))}
                      className="w-full accent-teal-500 bg-slate-950 h-1.5 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">Chunk Overlap</span>
                      <span className="text-violet-400 font-mono font-bold">{chunkOverlap}</span>
                    </div>
                    <input 
                      type="range" min="0" max="300" step="10"
                      value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))}
                      className="w-full accent-violet-500 bg-slate-950 h-1.5 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </div>

                {/* Chunk List Previews */}
                <div className="mt-4 pt-4 border-t border-white/5 space-y-2.5">
                  <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Generated Chunks ({chunksPreview.length})</div>
                  <div className="max-h-48 overflow-y-auto space-y-2 pr-1">
                    {chunksPreview.map((c, i) => (
                      <div key={i} className="bg-slate-950/60 border border-white/5 rounded-lg p-2.5 text-xs">
                        <div className="flex justify-between text-[9px] text-slate-500 font-bold mb-1">
                          <span>CHUNK #{i + 1}</span>
                          <span>{c.token_count} TOKENS</span>
                        </div>
                        <p className="text-slate-300 leading-relaxed italic">"{c.text_content}"</p>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            </div>

            {/* Right side: Retrieval compare, Playgrounds, Reranking & Chat */}
            <div className="lg:col-span-8 space-y-6">
              
              {/* Vector DB & Embeddings specs */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Embeddings Provider details */}
                <div className="glass-panel rounded-2xl p-5 shadow-elegant">
                  <h3 className="text-sm font-bold mb-3 text-teal-400 flex items-center">
                    <Cpu className="mr-2 h-4.5 w-4.5" /> Embedding Studio
                  </h3>
                  <div className="grid grid-cols-4 gap-2 mb-4">
                    {['google', 'openai', 'voyage', 'jina'].map(p => (
                      <button 
                        key={p} 
                        onClick={() => setSelectedEmbedder(p)}
                        className={`py-1.5 text-xs font-bold rounded-lg border transition ${
                          selectedEmbedder === p 
                            ? 'bg-teal-500/10 border-teal-500/40 text-teal-300' 
                            : 'border-white/5 hover:border-white/10 text-slate-400'
                        }`}
                      >
                        {p.toUpperCase()}
                      </button>
                    ))}
                  </div>

                  <div className="bg-slate-950/40 rounded-xl p-3 border border-white/5 space-y-2 text-xs">
                    <div className="flex justify-between"><span className="text-slate-400">Model:</span><span className="font-semibold">{selectedEmbedder === 'google' ? 'text-embedding-004' : selectedEmbedder === 'openai' ? 'text-embedding-3-small' : 'voyage-2'}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Dimensions:</span><span className="font-mono text-teal-400 font-bold">{selectedEmbedder === 'google' ? 768 : selectedEmbedder === 'openai' ? 1536 : 1024}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">MTEB Leaderboard Score:</span><span className="font-semibold">{selectedEmbedder === 'google' ? '66.2' : selectedEmbedder === 'openai' ? '64.5' : '68.1'}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Ingestion Cost / 1k:</span><span className="font-semibold text-violet-400">$0.00004</span></div>
                  </div>
                </div>

                {/* Vector Database specs */}
                <div className="glass-panel rounded-2xl p-5 shadow-elegant">
                  <h3 className="text-sm font-bold mb-3 text-violet-400 flex items-center">
                    <Database className="mr-2 h-4.5 w-4.5" /> Vector Database
                  </h3>
                  <div className="grid grid-cols-3 gap-2 mb-4">
                    {['qdrant', 'pinecone', 'pgvector'].map(db => (
                      <button 
                        key={db} 
                        onClick={() => setSelectedVectorDb(db)}
                        className={`py-1.5 text-xs font-bold rounded-lg border transition ${
                          selectedVectorDb === db 
                            ? 'bg-violet-500/10 border-violet-500/40 text-violet-300' 
                            : 'border-white/5 hover:border-white/10 text-slate-400'
                        }`}
                      >
                        {db.toUpperCase()}
                      </button>
                    ))}
                  </div>

                  <div className="bg-slate-950/40 rounded-xl p-3 border border-white/5 space-y-2 text-xs">
                    <div className="flex justify-between"><span className="text-slate-400">Storage Index:</span><span className="font-semibold">HNSW + Cosine</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Active Namespaces:</span><span className="font-semibold">default_idx</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Index Memory Usage:</span><span className="font-mono text-teal-400 font-bold">12 MB</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Total Vectors Loaded:</span><span className="font-semibold">17 chunks</span></div>
                  </div>
                </div>
              </div>

              {/* Hybrid Search Tuner & RRF Visualizer */}
              <section className="glass-panel rounded-2xl p-5 shadow-elegant">
                <h3 className="text-base font-bold mb-3 flex items-center text-teal-400">
                  <Sliders className="mr-2 h-5 w-5" /> Hybrid Search & Fusion weights
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4 items-center">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400 font-semibold">Dense Vector (Semantic) Weight</span>
                      <span className="text-teal-400 font-bold">{denseWeight.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" min="0" max="1" step="0.05"
                      value={denseWeight} onChange={(e) => {
                        const val = Number(e.target.value);
                        setDenseWeight(val);
                        setSparseWeight(1 - val);
                      }}
                      className="w-full accent-teal-500 bg-slate-950 h-1.5 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400 font-semibold">Sparse Keyword (BM25) Weight</span>
                      <span className="text-violet-400 font-bold">{sparseWeight.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" min="0" max="1" step="0.05"
                      value={sparseWeight} onChange={(e) => {
                        const val = Number(e.target.value);
                        setSparseWeight(val);
                        setDenseWeight(1 - val);
                      }}
                      className="w-full accent-violet-500 bg-slate-950 h-1.5 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </div>

                {/* Compare Search input */}
                <div className="flex gap-2 mb-4">
                  <input 
                    type="text" 
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter test query..."
                    className="bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm flex-1 outline-none focus:border-teal-500"
                  />
                  <button 
                    onClick={runMockSearch}
                    disabled={isSearching}
                    className="bg-slate-800 hover:bg-slate-700 border border-white/10 text-slate-200 px-4 py-2 rounded-lg text-sm font-semibold flex items-center"
                  >
                    <Search className="h-4 w-4 mr-2" /> {isSearching ? 'Comparing...' : 'Compare Search'}
                  </button>
                </div>

                {/* Compare side-by-side search results */}
                {searchResults && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-white/5">
                    
                    {/* Dense Search Result column */}
                    <div className="bg-slate-950/40 border border-white/5 rounded-xl p-3 space-y-2">
                      <span className="text-[10px] font-bold text-teal-400 uppercase tracking-widest">Dense (Similarity)</span>
                      {searchResults.dense.map((r: any, i: number) => (
                        <div key={i} className="text-xs p-2 bg-slate-900 border border-white/5 rounded-lg">
                          <div className="flex justify-between text-[9px] text-slate-500 font-bold mb-1">
                            <span>RANK {i + 1}</span>
                            <span>SCORE {r.score}</span>
                          </div>
                          <p className="text-slate-300 italic">"{r.text.slice(0, 50)}..."</p>
                        </div>
                      ))}
                    </div>

                    {/* Sparse Search Result column */}
                    <div className="bg-slate-950/40 border border-white/5 rounded-xl p-3 space-y-2">
                      <span className="text-[10px] font-bold text-violet-400 uppercase tracking-widest">Sparse (BM25)</span>
                      {searchResults.sparse.map((r: any, i: number) => (
                        <div key={i} className="text-xs p-2 bg-slate-900 border border-white/5 rounded-lg">
                          <div className="flex justify-between text-[9px] text-slate-500 font-bold mb-1">
                            <span>RANK {i + 1}</span>
                            <span>SCORE {r.score}</span>
                          </div>
                          <p className="text-slate-300 italic">"{r.text.slice(0, 50)}..."</p>
                        </div>
                      ))}
                    </div>

                    {/* Reciprocal Rank Fused Result column */}
                    <div className="bg-slate-950/40 border border-teal-500/20 rounded-xl p-3 space-y-2 shadow-glow">
                      <span className="text-[10px] font-bold text-teal-300 uppercase tracking-widest flex items-center">
                        <Sparkles className="h-3 w-3 mr-1 text-teal-400" /> Fused (RRF Hybrid)
                      </span>
                      {searchResults.fused.map((r: any, i: number) => (
                        <div key={i} className="text-xs p-2 bg-teal-500/5 border border-teal-500/10 rounded-lg">
                          <div className="flex justify-between text-[9px] text-slate-400 font-bold mb-1">
                            <span>RANK {r.rank}</span>
                            <span>HYBRID SCORE {r.score}</span>
                          </div>
                          <p className="text-slate-200">"{r.text.slice(0, 55)}..."</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>

              {/* Chat Playground & Explainability */}
              <section className="glass-panel rounded-2xl p-5 shadow-elegant flex flex-col h-[400px]">
                <div className="flex justify-between items-center mb-4 pb-2 border-b border-white/5">
                  <h3 className="text-base font-bold flex items-center text-teal-400">
                    <Terminal className="mr-2 h-5 w-5" /> Chat Playground
                  </h3>
                  
                  {/* LLM Model Select */}
                  <select 
                    value={selectedLlm} 
                    onChange={(e) => setSelectedLlm(e.target.value)}
                    className="bg-slate-900 border border-white/10 rounded px-2.5 py-1 text-xs outline-none text-violet-400 font-semibold"
                  >
                    <option value="google-gemini-flash">Gemini 2.5 Flash</option>
                    <option value="openai-gpt-4o">GPT-4o</option>
                    <option value="anthropic-claude">Claude 3.5 Sonnet</option>
                    <option value="deepseek-r1">DeepSeek R1</option>
                  </select>
                </div>

                {/* Message Log */}
                <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-1">
                  {chatHistory.map((msg, index) => (
                    <div key={index} className="space-y-2">
                      <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                          msg.role === 'user' 
                            ? 'bg-gradient-to-r from-teal-500 to-violet-500 text-white' 
                            : 'bg-white/5 border border-white/5 text-slate-200'
                        }`}>
                          <p>{msg.text}</p>
                        </div>
                      </div>

                      {/* Explainability Accordion */}
                      {msg.role === 'agent' && msg.trace && (
                        <div className="pl-4">
                          <button 
                            onClick={() => setShowTraceForIndex(showTraceForIndex === index ? null : index)}
                            className="text-[10px] text-teal-400 hover:text-teal-300 font-bold uppercase tracking-wider flex items-center space-x-1"
                          >
                            <span>{showTraceForIndex === index ? '▼ Hide explainability trace' : '► Explain retrieval steps'}</span>
                          </button>

                          {showTraceForIndex === index && (
                            <div className="mt-2 bg-slate-950/60 border border-white/5 rounded-xl p-4 text-xs space-y-3 max-w-[90%]">
                              <div>
                                <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">Retrieved ground chunks</div>
                                <div className="space-y-2">
                                  {msg.trace.retrieved.map((ch: any, cidx: number) => (
                                    <div key={cidx} className="bg-slate-900/60 p-2 rounded-lg border border-white/5">
                                      <div className="flex justify-between text-[9px] text-slate-500 mb-1">
                                        <span>Rank #{ch.rank} ({ch.source})</span>
                                        <span className="text-teal-400">Score: {ch.similarity}</span>
                                      </div>
                                      <p className="text-slate-300 italic">"{ch.text}"</p>
                                    </div>
                                  ))}
                                </div>
                              </div>

                              <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5 text-[10px]">
                                <div>
                                  <span className="text-slate-500 block">RERANK MODEL</span>
                                  <span className="font-semibold text-slate-300">{msg.trace.rerankModel}</span>
                                </div>
                                <div>
                                  <span className="text-slate-500 block">LATENCY</span>
                                  <span className="font-semibold text-slate-300">{msg.trace.latencyMs} ms</span>
                                </div>
                                <div>
                                  <span className="text-slate-500 block">HALLUCINATION RISK</span>
                                  <span className="font-semibold text-teal-400">{(msg.trace.hallucinationIndex * 100).toFixed(0)}% (Very Low)</span>
                                </div>
                                <div>
                                  <span className="text-slate-500 block">HYBRID WEIGHTS</span>
                                  <span className="font-semibold text-slate-300">Dense {msg.trace.denseWeight} / Sparse {msg.trace.sparseWeight}</span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Input row */}
                <form onSubmit={handleSendMessage} className="flex gap-2 border-t border-white/5 pt-3">
                  <input 
                    type="text" 
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask a question about the loaded context..."
                    className="bg-slate-950 border border-white/10 rounded-lg px-4 py-2.5 text-sm flex-1 outline-none focus:border-teal-500"
                  />
                  <button 
                    type="submit" 
                    className="bg-gradient-to-r from-teal-500 to-violet-500 text-white p-2.5 rounded-lg hover:brightness-110 active:scale-95 transition"
                  >
                    <Send className="h-5 w-5" />
                  </button>
                </form>
              </section>

            </div>
          </div>
        )}

        {/* TAB 2: EVALUATE STUDIO */}
        {activeTab === 'evaluate' && (
          <div className="space-y-6 animate-fade-in">
            
            {/* Side-by-side Experiment comparisons */}
            <section className="glass-panel rounded-2xl p-6 shadow-elegant">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-lg font-bold text-teal-400">Evaluation Studio</h3>
                  <p className="text-xs text-slate-400">Compare pipeline experiments and retrieval quality side-by-side</p>
                </div>
                <button 
                  onClick={() => {
                    const newRun = {
                      name: `Experiment ${evalRuns.length + 1} (Semantic)`,
                      strategy: "Semantic",
                      size: 600,
                      overlap: 0,
                      recall: 0.92,
                      faithfulness: 0.95,
                      cost: 0.009,
                      latency: 280
                    };
                    setEvalRuns([...evalRuns, newRun]);
                  }}
                  className="bg-gradient-to-r from-teal-500 to-violet-500 text-white px-4 py-2 rounded-lg text-sm font-bold hover:brightness-110 active:scale-95 transition flex items-center"
                >
                  <Play className="h-4 w-4 mr-2" /> Trigger New Evaluation Run
                </button>
              </div>

              {/* Evaluation Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-white/10 text-xs text-slate-500 uppercase font-bold">
                      <th className="py-3 px-4">Pipeline Experiment Name</th>
                      <th className="py-3 px-4">Chunk Strategy</th>
                      <th className="py-3 px-4">Chunk Size</th>
                      <th className="py-3 px-4 text-center">Recall@K</th>
                      <th className="py-3 px-4 text-center">Faithfulness</th>
                      <th className="py-3 px-4 text-center">Avg Latency</th>
                      <th className="py-3 px-4 text-center">LLM Cost / Query</th>
                      <th className="py-3 px-4 text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm divide-y divide-white/5">
                    {evalRuns.map((run, i) => (
                      <tr key={i} className="hover:bg-white/5 transition-colors">
                        <td className="py-3.5 px-4 font-semibold text-slate-200">{run.name}</td>
                        <td className="py-3.5 px-4 text-slate-400">{run.strategy}</td>
                        <td className="py-3.5 px-4 text-slate-400 font-mono">{run.size} chars</td>
                        <td className="py-3.5 px-4 text-center">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                            run.recall >= 0.85 ? 'bg-teal-500/10 text-teal-400' : 'bg-amber-500/10 text-amber-400'
                          }`}>{run.recall * 100}%</span>
                        </td>
                        <td className="py-3.5 px-4 text-center">
                          <span className="px-2 py-0.5 rounded text-xs font-bold bg-violet-500/10 text-violet-400">{run.faithfulness * 100}%</span>
                        </td>
                        <td className="py-3.5 px-4 text-center font-mono text-slate-300">{run.latency} ms</td>
                        <td className="py-3.5 px-4 text-center font-mono text-slate-300">${run.cost.toFixed(3)}</td>
                        <td className="py-3.5 px-4 text-right">
                          <span className="bg-teal-500/10 text-teal-400 border border-teal-500/20 text-[10px] font-bold px-2 py-0.5 rounded-full">
                            COMPLETED
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Knowledge Graph Entities Extractor Visualization */}
            <section className="glass-panel rounded-2xl p-6 shadow-elegant">
              <h3 className="text-base font-bold mb-2 text-violet-400 flex items-center">
                <Network className="mr-2 h-5 w-5" /> Module 18: Knowledge Graph Extractor
              </h3>
              <p className="text-xs text-slate-400 mb-6">Interactive entity links automatically extracted from collections</p>
              
              <div className="bg-slate-950/60 rounded-2xl p-6 border border-white/5 flex items-center justify-center min-h-[220px] relative overflow-hidden">
                
                {/* SVG mock graph */}
                <svg className="w-full max-w-lg h-48" viewBox="0 0 400 200">
                  {/* Entity links */}
                  <line x1="80" y1="100" x2="200" y2="60" stroke="rgba(6, 182, 212, 0.4)" strokeWidth="2" />
                  <line x1="80" y1="100" x2="200" y2="140" stroke="rgba(6, 182, 212, 0.4)" strokeWidth="2" />
                  <line x1="200" y1="60" x2="320" y2="100" stroke="rgba(139, 92, 246, 0.4)" strokeWidth="2" />
                  <line x1="200" y1="140" x2="320" y2="100" stroke="rgba(139, 92, 246, 0.4)" strokeWidth="2" />
                  <line x1="200" y1="60" x2="200" y2="140" stroke="rgba(255, 255, 255, 0.1)" strokeWidth="1" strokeDasharray="4" />

                  {/* Entity nodes */}
                  <circle cx="80" cy="100" r="24" fill="rgba(6, 182, 212, 0.2)" stroke="#06b6d4" strokeWidth="2" />
                  <text x="80" y="104" fill="#06b6d4" fontSize="9" fontWeight="bold" textAnchor="middle">RAG</text>

                  <circle cx="200" cy="60" r="28" fill="rgba(139, 92, 246, 0.2)" stroke="#8b5cf6" strokeWidth="2" />
                  <text x="200" y="64" fill="#8b5cf6" fontSize="9" fontWeight="bold" textAnchor="middle">Embeddings</text>

                  <circle cx="200" cy="140" r="28" fill="rgba(139, 92, 246, 0.2)" stroke="#8b5cf6" strokeWidth="2" />
                  <text x="200" y="144" fill="#8b5cf6" fontSize="9" fontWeight="bold" textAnchor="middle">Vector DB</text>

                  <circle cx="320" cy="100" r="24" fill="rgba(6, 182, 212, 0.2)" stroke="#06b6d4" strokeWidth="2" />
                  <text x="320" y="104" fill="#06b6d4" fontSize="9" fontWeight="bold" textAnchor="middle">Context</text>
                </svg>
                
                <div className="absolute bottom-4 right-4 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-white/5 text-[10px] text-slate-400">
                  Showing 4 entities and 5 extracted relationships
                </div>
              </div>
            </section>
          </div>
        )}

        {/* TAB 3: MONITOR & ANALYTICS */}
        {activeTab === 'monitor' && (
          <div className="space-y-6 animate-fade-in">
            
            {/* KPI metric grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              
              <div className="glass-panel rounded-2xl p-5 shadow-elegant">
                <div className="flex justify-between items-center text-slate-400 mb-2">
                  <span className="text-xs font-semibold">Total Queries</span>
                  <Activity className="h-5 w-5 text-teal-400" />
                </div>
                <div className="text-2xl font-black text-slate-200">1,824</div>
                <div className="text-[10px] text-teal-400 font-bold mt-1">▲ 14% vs last week</div>
              </div>

              <div className="glass-panel rounded-2xl p-5 shadow-elegant">
                <div className="flex justify-between items-center text-slate-400 mb-2">
                  <span className="text-xs font-semibold">Avg Retrieval Latency</span>
                  <Sliders className="h-5 w-5 text-violet-400" />
                </div>
                <div className="text-2xl font-black text-slate-200">184 ms</div>
                <div className="text-[10px] text-teal-400 font-bold mt-1">▼ 12 ms reduction</div>
              </div>

              <div className="glass-panel rounded-2xl p-5 shadow-elegant">
                <div className="flex justify-between items-center text-slate-400 mb-2">
                  <span className="text-xs font-semibold">Total API Costs</span>
                  <Cpu className="h-5 w-5 text-teal-400" />
                </div>
                <div className="text-2xl font-black text-slate-200">$98.57</div>
                <div className="text-[10px] text-slate-400 mt-1">Google API $12.45 · LLM $86.12</div>
              </div>

              <div className="glass-panel rounded-2xl p-5 shadow-elegant">
                <div className="flex justify-between items-center text-slate-400 mb-2">
                  <span className="text-xs font-semibold">Cache Hit Rate</span>
                  <Database className="h-5 w-5 text-violet-400" />
                </div>
                <div className="text-2xl font-black text-slate-200">24%</div>
                <div className="text-[10px] text-violet-400 font-bold mt-1">Saved 420 calls to LLM</div>
              </div>

            </div>

            {/* In-depth cost & feedback charts */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              
              <div className="lg:col-span-8 glass-panel rounded-2xl p-6 shadow-elegant">
                <h3 className="text-base font-bold mb-4 text-teal-400">Daily Query Analytics</h3>
                
                {/* Simulated bar chart */}
                <div className="h-48 flex items-end justify-between pt-6 border-b border-white/10">
                  {[45, 60, 52, 78, 88, 92, 110, 95, 120, 150].map((val, idx) => (
                    <div key={idx} className="w-[8%] flex flex-col items-center">
                      <div 
                        className="w-full bg-gradient-to-t from-teal-500/40 to-teal-500 rounded-t-md hover:brightness-110 transition-all duration-300"
                        style={{ height: `${val}%` }}
                      />
                      <span className="text-[9px] text-slate-500 mt-2">D{idx+1}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Human feedback loop & telemetry */}
              <div className="lg:col-span-4 glass-panel rounded-2xl p-6 shadow-elegant space-y-4">
                <h3 className="text-base font-bold text-violet-400">Telemetry & User Feedback</h3>
                
                <div className="bg-slate-950/60 border border-white/5 rounded-xl p-4 flex items-center justify-between">
                  <div>
                    <div className="text-xs text-slate-400 font-semibold">Thumbs Up Rating</div>
                    <div className="text-xl font-extrabold text-slate-200 mt-1">94.8%</div>
                  </div>
                  <div className="bg-teal-500/10 p-2.5 rounded-full border border-teal-500/20">
                    <ThumbsUp className="h-5 w-5 text-teal-400" />
                  </div>
                </div>

                <div className="bg-slate-950/60 border border-white/5 rounded-xl p-4 flex items-center justify-between">
                  <div>
                    <div className="text-xs text-slate-400 font-semibold">Flagged Hallucinations</div>
                    <div className="text-xl font-extrabold text-slate-200 mt-1">2 queries</div>
                  </div>
                  <div className="bg-amber-500/10 p-2.5 rounded-full border border-amber-500/20">
                    <ShieldAlert className="h-5 w-5 text-amber-400" />
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

      </main>

      {/* Footer copyright */}
      <footer className="py-6 text-center text-xs text-slate-500 border-t border-white/5 mt-auto">
        <p>© 2026 RAG Studio. Powered by Google Gemini & Qdrant.</p>
      </footer>

    </div>
  );
}
