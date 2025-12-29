
import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI, Modality } from "@google/genai";

// --- Types & Interfaces ---
type Role = "user" | "model";
type SchemeType = "vision" | "intel" | "entry" | "design" | "tech";

interface Message {
  role: Role;
  text: string;
  isError?: boolean;
  timestamp: number;
  imageUrl?: string;
  grounding?: any[];
  audioData?: string;
}

interface Scheme {
  id: SchemeType;
  name: string;
  tagline: string;
  icon: React.ReactNode;
  instruction: string;
  color: string;
  tool?: "search" | "image" | "maps";
}

interface ProjectContext {
  company: string;
  industry: string;
  market: string;
}

// --- Audio Utilities ---
const decodeBase64 = (base64: string) => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  return bytes;
};

const decodeAudioData = async (data: Uint8Array, ctx: AudioContext): Promise<AudioBuffer> => {
  const dataInt16 = new Int16Array(data.buffer);
  const buffer = ctx.createBuffer(1, dataInt16.length, 24000);
  const channelData = buffer.getChannelData(0);
  for (let i = 0; i < dataInt16.length; i++) channelData[i] = dataInt16[i] / 32768.0;
  return buffer;
};

// --- Strategic Presets ---
const SCHEMES: Scheme[] = [
  { 
    id: "vision", 
    name: "Vision Architect", 
    tagline: "Expansion Moats",
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>, 
    instruction: "You are the Vision Architect. Focus on long-term competitive moats, SWOT frameworks, and PESTLE analysis for expansion.",
    color: "from-blue-600/30 to-indigo-600/30"
  },
  { 
    id: "intel", 
    name: "Market Intel", 
    tagline: "Live Web Intel",
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>, 
    instruction: "You are a Market Intel Analyst. Use Google Search to find real-time trends, pricing, and competitor shifts.",
    color: "from-emerald-600/30 to-teal-600/30",
    tool: "search"
  },
  { 
    id: "entry", 
    name: "Global Entry", 
    tagline: "Maps & Logistics",
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>, 
    instruction: "You are a Global Expansion Logistics Expert. Use Google Maps to identify target locations, business hotspots, and transit routes.",
    color: "from-orange-600/30 to-amber-600/30",
    tool: "maps"
  },
  { 
    id: "design", 
    name: "Brand Design", 
    tagline: "Visual Prototypes",
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>, 
    instruction: "You are a Brand Designer. Generate visuals and UI mockups for new market branding.",
    color: "from-rose-600/30 to-pink-600/30",
    tool: "image"
  },
  { 
    id: "tech", 
    name: "Tech Architect", 
    tagline: "Scalable Systems",
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>, 
    instruction: "You are a CTO. Focus on cloud scaling, localized data compliance, and high-availability architecture.",
    color: "from-purple-600/30 to-violet-600/30"
  },
];

// --- Components ---

const CitationLink = ({ chunk }: { chunk: any }) => {
  const uri = chunk.web?.uri || chunk.maps?.uri;
  const title = chunk.web?.title || chunk.maps?.title || "Resource";
  if (!uri) return null;
  return (
    <a href={uri} target="_blank" rel="noopener" className="inline-flex items-center gap-1.5 px-2 py-1 bg-white/5 border border-white/5 rounded-lg text-[10px] text-cyan-400 hover:bg-white/10 transition-colors group">
      <svg className="w-3 h-3 text-slate-500 group-hover:text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" /></svg>
      {title.length > 25 ? title.substring(0, 25) + "..." : title}
    </a>
  );
};

const App = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeScheme, setActiveScheme] = useState<SchemeType>("vision");
  const [isListening, setIsListening] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [showIntel, setShowIntel] = useState(true);
  const [project, setProject] = useState<ProjectContext>({ company: "Chi AI", industry: "SaaS Expansion", market: "Global" });

  const scrollRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRec) {
      const rec = new SpeechRec();
      rec.onresult = (e: any) => setInputValue(e.results[0][0].transcript);
      rec.onend = () => setIsListening(false);
      recognitionRef.current = rec;
    }
  }, []);

  const speak = async (text: string, index?: number) => {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || "" });
    setIsSpeaking(true);
    try {
      const resp = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: `Executive Briefing: ${text.substring(0, 800)}` }] }],
        config: { responseModalities: [Modality.AUDIO], speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } } }
      });
      const b64 = resp.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (b64) {
        if (!audioContextRef.current) audioContextRef.current = new AudioContext({ sampleRate: 24000 });
        const buf = await decodeAudioData(decodeBase64(b64), audioContextRef.current);
        const src = audioContextRef.current.createBufferSource();
        src.buffer = buf;
        src.connect(audioContextRef.current.destination);
        src.onended = () => setIsSpeaking(false);
        src.start();
        if (index !== undefined) {
          setMessages(prev => {
            const next = [...prev];
            next[index].audioData = b64;
            return next;
          });
        }
      }
    } catch { setIsSpeaking(false); }
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;
    const userText = inputValue;
    const currentScheme = SCHEMES.find(s => s.id === activeScheme)!;
    setInputValue("");
    setMessages(prev => [...prev, { role: "user", text: userText, timestamp: Date.now() }]);
    setIsLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || "" });
      const systemInstruction = `${currentScheme.instruction}. Context: We are working for ${project.company} in the ${project.industry} industry, targeting the ${project.market} market. Provide senior-level strategic advice.`;
      
      let res: any;
      if (currentScheme.tool === "image") {
        res = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image',
          contents: { parts: [{ text: userText }] },
          config: { imageConfig: { aspectRatio: "16:9" } }
        });
      } else {
        const config: any = { systemInstruction, thinkingConfig: { thinkingBudget: 16000 } };
        if (currentScheme.tool === "search") config.tools = [{ googleSearch: {} }];
        if (currentScheme.tool === "maps") {
          config.tools = [{ googleMaps: {} }];
          try {
            const pos: any = await new Promise((res, rej) => navigator.geolocation.getCurrentPosition(res, rej));
            config.toolConfig = { retrievalConfig: { latLng: { latitude: pos.coords.latitude, longitude: pos.coords.longitude } } };
          } catch {}
        }

        const chat = ai.chats.create({ model: "gemini-3-pro-preview", config, history: messages.slice(-6).map(m => ({ role: m.role, parts: [{ text: m.text }] })) });
        res = await chat.sendMessage({ message: userText });
      }

      let foundText = res.text || "";
      let foundImage = "";
      const parts = res.candidates?.[0]?.content?.parts || [];
      for (const p of parts) {
        if (p.inlineData) foundImage = `data:image/png;base64,${p.inlineData.data}`;
      }

      const grounding = res.candidates?.[0]?.groundingMetadata?.groundingChunks;
      const newMessage: Message = { role: "model", text: foundText, imageUrl: foundImage, grounding, timestamp: Date.now() };
      setMessages(prev => [...prev, newMessage]);
      if (autoSpeak && foundText) speak(foundText);
    } catch (err: any) {
      setMessages(prev => [...prev, { role: "model", text: `Strategic Interrupt: ${err.message}`, isError: true, timestamp: Date.now() }]);
    } finally { setIsLoading(false); }
  };

  const activeSchemeData = SCHEMES.find(s => s.id === activeScheme)!;

  return (
    <div className="flex h-screen bg-[#030306] text-slate-300 font-sans selection:bg-cyan-500/30 selection:text-white">
      
      {/* Mesh Gradients */}
      <div className={`fixed inset-0 opacity-20 bg-gradient-to-br ${activeSchemeData.color} blur-[120px] transition-all duration-1000`}></div>
      <div className="fixed inset-0 bg-[radial-gradient(circle_at_50%_-20%,rgba(120,119,198,0.1),transparent)]"></div>

      {/* Sidebar: Strategic Navigation */}
      <aside className="w-80 h-full glass-panel border-r border-white/5 flex flex-col z-30 shrink-0">
        <div className="p-8 pb-4">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <span className="text-white font-black text-lg">χ</span>
            </div>
            <div>
              <h1 className="text-base font-bold text-white tracking-tight">Chi Strategy</h1>
              <p className="text-[9px] font-mono text-cyan-400 uppercase tracking-widest opacity-70">Enterprise MVP v3.1</p>
            </div>
          </div>

          <div className="space-y-1.5">
            <p className="text-[10px] font-mono text-slate-500 uppercase tracking-[0.2em] mb-4 pl-2">Expansion Modules</p>
            {SCHEMES.map(s => (
              <button
                key={s.id}
                onClick={() => setActiveScheme(s.id)}
                className={`w-full flex items-center gap-3.5 p-3 rounded-2xl transition-all group border ${
                  activeScheme === s.id ? "bg-white/10 border-white/10 text-white" : "border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5"
                }`}
              >
                <div className={`w-9 h-9 rounded-xl flex items-center justify-center transition-colors ${activeScheme === s.id ? "bg-cyan-500/20 text-cyan-400" : "bg-white/5"}`}>
                  {s.icon}
                </div>
                <div className="text-left">
                  <p className="text-[13px] font-bold leading-none mb-1">{s.name}</p>
                  <p className="text-[10px] opacity-50">{s.tagline}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Project Context Panel */}
        <div className={`mt-6 mx-4 p-5 rounded-2xl bg-white/5 border border-white/5 transition-all ${showIntel ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'}`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Project Intel</h3>
            <button onClick={() => setShowIntel(false)} className="text-slate-600 hover:text-slate-400"><svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M6 18L18 6M6 6l12 12" strokeWidth={2}/></svg></button>
          </div>
          <div className="space-y-3">
            {[
              { label: 'COMPANY', key: 'company' as const },
              { label: 'INDUSTRY', key: 'industry' as const },
              { label: 'MARKET', key: 'market' as const }
            ].map(f => (
              <div key={f.key}>
                <label className="text-[9px] font-mono text-slate-600 block mb-1">{f.label}</label>
                <input 
                  type="text" 
                  value={project[f.key]}
                  onChange={e => setProject({...project, [f.key]: e.target.value})}
                  className="w-full bg-black/30 border border-white/5 rounded-lg px-2.5 py-1.5 text-[11px] text-slate-300 outline-none focus:border-cyan-500/40 transition-colors"
                />
              </div>
            ))}
          </div>
        </div>

        <div className="mt-auto p-6 border-t border-white/5">
          <button 
            onClick={() => setAutoSpeak(!autoSpeak)}
            className={`w-full p-3 rounded-xl flex items-center justify-between text-[11px] font-bold transition-all border ${autoSpeak ? 'bg-cyan-500/10 border-cyan-500/20 text-cyan-400' : 'bg-white/5 border-transparent text-slate-500'}`}
          >
            <span>VOICE PROTOCOL</span>
            <div className={`w-2 h-2 rounded-full ${autoSpeak ? 'bg-cyan-500 animate-pulse' : 'bg-slate-700'}`}></div>
          </button>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col relative z-20 overflow-hidden">
        <header className="h-20 flex items-center justify-between px-10 border-b border-white/5 backdrop-blur-xl">
          <div className="flex items-center gap-4">
            {!showIntel && (
              <button onClick={() => setShowIntel(true)} className="p-2 bg-white/5 rounded-lg text-slate-500 hover:text-cyan-400 transition-colors">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M4 6h16M4 12h16m-7 6h7" strokeWidth={2}/></svg>
              </button>
            )}
            <div>
              <h2 className="text-sm font-bold text-white uppercase tracking-widest">{activeSchemeData.name}</h2>
              <div className="flex items-center gap-2 text-[10px] text-slate-500">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                ACTIVE STRATEGY: {project.company}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            {isSpeaking && (
              <div className="flex items-end gap-1 h-3 pb-0.5">
                {[...Array(6)].map((_, i) => <div key={i} className="w-0.5 bg-cyan-500 rounded-full animate-[pulse_0.6s_infinite_ease-in-out]" style={{ animationDelay: `${i*0.1}s` }}></div>)}
              </div>
            )}
            <div className="px-3 py-1 rounded-full bg-white/5 border border-white/5 text-[10px] font-mono text-slate-500">PRO-NODE: 09-2025</div>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-10 space-y-10 scroll-smooth">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center">
              <div className="text-center max-w-lg">
                <div className={`w-20 h-20 mx-auto mb-8 rounded-[2rem] bg-gradient-to-br ${activeSchemeData.color} flex items-center justify-center text-white shadow-2xl border border-white/10`}>
                  {activeSchemeData.icon}
                </div>
                <h2 className="text-3xl font-black text-white mb-4">Initialize Growth</h2>
                <p className="text-slate-400 text-sm leading-relaxed mb-10">
                  Select a strategic module and define your project context to begin. Powered by Gemini 3 Pro and live Google Grounding.
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {["Analyze competitor density in London", "Logo concepts for MedGemm", "Cloud architecture SWOT", "Market entry barriers"].map(t => (
                    <button key={t} onClick={() => setInputValue(t)} className="px-4 py-2 rounded-xl bg-white/5 border border-white/5 text-[11px] text-slate-400 hover:bg-white/10 hover:text-white transition-all">
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-12">
              {messages.map((m, i) => (
                <div key={i} className={`flex gap-6 ${m.role === 'user' ? 'flex-row-reverse' : ''} animate-[fadeIn_0.5s_ease-out]`}>
                  <div className={`w-10 h-10 rounded-2xl shrink-0 flex items-center justify-center border border-white/10 shadow-xl ${m.role === 'user' ? 'bg-indigo-600/30 text-indigo-300' : 'bg-slate-800 text-cyan-400'}`}>
                    {m.role === 'user' ? <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" strokeWidth={2}/></svg> : <span className="font-black text-lg">χ</span>}
                  </div>
                  <div className={`max-w-[85%] space-y-2 ${m.role === 'user' ? 'text-right' : ''}`}>
                    <div className={`p-7 rounded-[2.5rem] border shadow-2xl text-sm leading-relaxed ${m.role === 'user' ? 'bg-indigo-500/10 border-indigo-500/20 rounded-tr-none' : 'bg-slate-900/60 border-white/5 rounded-tl-none backdrop-blur-md'}`}>
                      <div className="prose prose-invert prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: m.text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>').replace(/\n/g, '<br/>') }}></div>
                      
                      {m.imageUrl && <div className="mt-6 rounded-3xl overflow-hidden border border-white/10"><img src={m.imageUrl} alt="AI Gen" className="w-full h-auto" /></div>}
                      
                      {m.grounding && (
                        <div className="mt-6 pt-5 border-t border-white/10">
                          <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-3">Intelligence Sources</p>
                          <div className="flex flex-wrap gap-2">{m.grounding.map((c, ci) => <CitationLink key={ci} chunk={c} />)}</div>
                        </div>
                      )}
                    </div>
                    {m.role === 'model' && (
                      <div className="flex gap-4 px-4 opacity-50 hover:opacity-100 transition-opacity">
                        <button onClick={() => speak(m.text, i)} className="text-[10px] font-mono hover:text-cyan-400 flex items-center gap-1.5"><svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" strokeWidth={2}/></svg> REPLAY BRIEF</button>
                        <button onClick={() => navigator.clipboard.writeText(m.text)} className="text-[10px] font-mono hover:text-cyan-400">COPY MD</button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-6 animate-pulse">
                  <div className="w-10 h-10 rounded-2xl bg-slate-800 border border-white/10"></div>
                  <div className="p-5 bg-slate-900/40 rounded-[2rem] border border-white/5 w-64 h-12 flex items-center justify-center">
                    <div className="flex gap-1.5">
                      {[0, 1, 2].map(i => <div key={i} className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: `${i*0.1}s` }}></div>)}
                    </div>
                  </div>
                </div>
              )}
              <div ref={scrollRef} className="h-24" />
            </div>
          )}
        </div>

        {/* Strategic Dock */}
        <footer className="p-10 pt-0">
          <div className="max-w-4xl mx-auto relative group">
            <div className="absolute -inset-1 rounded-[32px] bg-gradient-to-r from-cyan-500 to-indigo-600 blur-xl opacity-20 group-hover:opacity-40 transition-opacity pointer-events-none"></div>
            <div className="relative glass-panel rounded-[28px] border border-white/10 p-2 flex items-end gap-2 shadow-2xl">
              <button 
                onClick={() => { setIsListening(!isListening); isListening ? recognitionRef.current?.stop() : recognitionRef.current?.start(); }}
                className={`p-4 rounded-2xl transition-all ${isListening ? 'bg-red-500/20 text-red-400 animate-pulse' : 'text-slate-500 hover:bg-white/5'}`}
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" strokeWidth={2}/></svg>
              </button>
              <textarea 
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                placeholder={isListening ? "Listening..." : `Prompt ${activeSchemeData.name} for ${project.company}...`}
                className="flex-1 bg-transparent border-none outline-none p-4 text-sm text-white placeholder-slate-600 resize-none min-h-[56px] max-h-40"
                rows={1}
              />
              <button 
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                className={`p-4 rounded-2xl transition-all ${!inputValue.trim() || isLoading ? 'text-slate-700' : 'bg-cyan-600 text-white shadow-lg shadow-cyan-600/30'}`}
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M13 7l5 5m0 0l-5 5m5-5H6" strokeWidth={2.5}/></svg>
              </button>
            </div>
            <div className="mt-4 flex justify-between px-6 text-[9px] font-mono text-slate-600 uppercase tracking-[0.2em]">
              <span>Grounding: {activeSchemeData.tool ? 'ACTIVE' : 'STATIC'}</span>
              <span className="text-cyan-600/60">Node: Gemini 3 Pro</span>
              <span>Encrypted Session</span>
            </div>
          </div>
        </footer>
      </main>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .glass-panel { background: rgba(10, 10, 18, 0.7); backdrop-filter: blur(32px); -webkit-backdrop-filter: blur(32px); }
        body { background: #030306; cursor: default; }
      `}</style>
    </div>
  );
};

createRoot(document.getElementById("root")!).render(<App />);
