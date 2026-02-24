import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Square, Play, BarChart3, MessageSquare, Quote, RefreshCw, Volume2 } from 'lucide-react';
import axios from 'axios';

const SCRIPT_TEXT = "Chào mừng quý vị và các bạn đã quay trở lại với chương trình The MC Hub ngày hôm nay. Tôi là người sẽ đồng hành cùng quý vị trong suốt hành trình khám phá những câu chuyện văn hóa đầy thú vị. Thưa quý vị, trong xu thế phát triển mạnh mẽ của công nghệ, việc giữ gìn bản sắc và hồn cốt của nghệ thuật truyền thống chính là một thách thức, nhưng cũng là cơ hội để chúng ta làm mới mình. Ngay bây giờ, mời quý vị hãy cùng chúng tôi bắt đầu bản tin với những tiêu điểm đáng chú ý nhất.";

function App() {
    const [isRecording, setIsRecording] = useState(false);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [activeWords, setActiveWords] = useState(new Set());
    const [status, setStatus] = useState('Sẵn sàng');

    const mediaRecorder = useRef(null);
    const audioChunks = useRef([]);
    const recognition = useRef(null);

    const words = SCRIPT_TEXT.split(' ');

    useEffect(() => {
        if ('webkitSpeechRecognition' in window) {
            const SpeechRecognition = window.webkitSpeechRecognition;
            recognition.current = new SpeechRecognition();
            recognition.current.continuous = true;
            recognition.current.interimResults = true;
            recognition.current.lang = 'vi-VN';

            recognition.current.onresult = (event) => {
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        const spoken = event.results[i][0].transcript.toLowerCase();
                        const spokenWords = spoken.split(' ');

                        setActiveWords(prev => {
                            const newSet = new Set(prev);
                            spokenWords.forEach(sWord => {
                                words.forEach((w, idx) => {
                                    const clean = w.toLowerCase().replace(/[.,!]/g, '');
                                    if (clean === sWord.trim()) {
                                        newSet.add(idx);
                                    }
                                });
                            });
                            return newSet;
                        });
                    }
                }
            };
        }
    }, []);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder.current = new MediaRecorder(stream);
            audioChunks.current = [];
            setActiveWords(new Set());
            setResult(null);

            mediaRecorder.current.ondataavailable = (e) => audioChunks.current.push(e.data);
            mediaRecorder.current.onstop = handleStop;

            mediaRecorder.current.start();
            if (recognition.current) recognition.current.start();
            setIsRecording(true);
            setStatus('Đang lắng nghe phong thái của bạn...');
        } catch (err) {
            alert('Không thể truy cập Microphone!');
        }
    };

    const stopRecording = () => {
        mediaRecorder.current.stop();
        if (recognition.current) recognition.current.stop();
        setIsRecording(false);
        setStatus('Đang phân tích...');
    };

    const handleStop = async () => {
        setLoading(true);
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('script_origin', SCRIPT_TEXT);

        try {
            const res = await axios.post('http://127.0.0.1:8000/analyze-voice', formData);
            setResult(res.data);
            setStatus('Hoàn tất phân tích');
        } catch (err) {
            setStatus('Lỗi kết nối Server');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-mesh py-12 px-4 selection:bg-blue-500/30">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <header className="text-center mb-16 space-y-4">
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="inline-flex items-center space-x-2 bg-blue-500/10 border border-blue-500/20 px-4 py-1.5 rounded-full"
                    >
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                        <span className="text-blue-400 text-sm font-medium tracking-wide uppercase">AI Coaching Engine</span>
                    </motion.div>
                    <motion.h1
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-6xl font-bold tracking-tight text-white mb-4"
                    >
                        The <span className="bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">MC Hub</span>
                    </motion.h1>
                    <p className="text-slate-400 text-lg max-w-xl mx-auto leading-relaxed">
                        Học viện luyện giọng nói MC ứng dụng trí tuệ nhân tạo. Phân tích phát âm, tốc độ và nhịp điệu chuyên sâu.
                    </p>
                </header>

                <main className="grid grid-cols-1 gap-8">
                    {/* Script Card */}
                    <section className="glass-card rounded-[32px] overflow-hidden">
                        <div className="p-8 border-b border-white/5 bg-white/5 flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                                <Quote className="w-5 h-5 text-blue-400" />
                                <h2 className="font-semibold text-white">Kịch bản bản tin văn hóa</h2>
                            </div>
                            <div className="text-xs text-slate-500 font-mono">LANG: VI-VN</div>
                        </div>
                        <div className="p-10">
                            <div className="relative">
                                <div className="absolute -left-4 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-500/50 to-transparent rounded-full" />
                                <div className="text-2xl leading-[1.8] text-slate-400 select-none">
                                    {words.map((word, idx) => (
                                        <span
                                            key={idx}
                                            className={`karaoke-word mr-2 ${activeWords.has(idx) ? 'active' : ''}`}
                                        >
                                            {word}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Control Bar */}
                    <div className="flex flex-col items-center space-y-6">
                        <div className="flex items-center space-x-8">
                            {!isRecording ? (
                                <button
                                    onClick={startRecording}
                                    disabled={loading}
                                    className="group relative flex items-center justify-center"
                                >
                                    <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-2xl group-hover:bg-blue-500/40 transition-all" />
                                    <div className="relative w-24 h-24 bg-gradient-to-tr from-blue-600 to-blue-400 rounded-full flex items-center justify-center shadow-2xl shadow-blue-500/30 group-hover:scale-105 transition-transform active:scale-95 disabled:grayscale">
                                        <Mic className="w-10 h-10 text-white" />
                                    </div>
                                </button>
                            ) : (
                                <button
                                    onClick={stopRecording}
                                    className="group relative flex items-center justify-center"
                                >
                                    <div className="absolute inset-0 bg-red-500/30 rounded-full blur-2xl animate-pulse" />
                                    <div className="relative w-24 h-24 bg-white rounded-3xl flex items-center justify-center shadow-2xl group-hover:scale-105 transition-transform active:scale-95">
                                        <Square className="w-10 h-10 text-red-500 fill-red-500" />
                                    </div>
                                </button>
                            )}
                        </div>
                        <p className="text-slate-500 font-medium tracking-wide">
                            {status}
                        </p>
                    </div>

                    {/* Results Area */}
                    <AnimatePresence>
                        {result && (
                            <motion.section
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="grid grid-cols-1 md:grid-cols-3 gap-6"
                            >
                                {/* Score Card */}
                                <div className="glass-card rounded-3xl p-8 text-center relative overflow-hidden group">
                                    <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/10 blur-[60px] group-hover:bg-green-500/20 transition-all" />
                                    <div className="relative">
                                        <BarChart3 className="w-6 h-6 text-green-400 mx-auto mb-4" />
                                        <div className="text-5xl font-bold text-green-400 mb-2">{result.accuracy_score}%</div>
                                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Độ chính xác</div>
                                    </div>
                                </div>

                                {/* Rhythm Card */}
                                <div className="glass-card rounded-3xl p-8 text-center relative overflow-hidden group">
                                    <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 blur-[60px] group-hover:bg-blue-500/20 transition-all" />
                                    <div className="relative">
                                        <Volume2 className="w-6 h-6 text-blue-400 mx-auto mb-4" />
                                        <div className="text-5xl font-bold text-blue-400 mb-2">{Math.round(result.speaking_rate_wpm)}</div>
                                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest">WPM (Tốc độ)</div>
                                    </div>
                                </div>

                                {/* Duration Card */}
                                <div className="glass-card rounded-3xl p-8 text-center relative overflow-hidden group">
                                    <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 blur-[60px] group-hover:bg-purple-500/20 transition-all" />
                                    <div className="relative">
                                        <BarChart3 className="w-6 h-6 text-purple-400 mx-auto mb-4" />
                                        <div className="text-5xl font-bold text-purple-400 mb-2">{result.duration_seconds}s</div>
                                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Thời lượng</div>
                                    </div>
                                </div>

                                {/* Expert Feedback */}
                                <div className="md:col-span-3 glass-card rounded-3xl p-10 bg-white/[0.02]">
                                    <div className="flex items-start space-x-6">
                                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                                            <MessageSquare className="w-8 h-8 text-white" />
                                        </div>
                                        <div className="space-y-4">
                                            <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                                                <span>Chuyên gia AI MC nhận xét</span>
                                                <div className="px-2 py-0.5 rounded-md bg-white/5 text-[10px] text-slate-500">BETA</div>
                                            </h3>
                                            <p className="text-lg text-slate-300 italic leading-relaxed">
                                                "{result.feedback}"
                                            </p>
                                            <div className="pt-4 border-t border-white/5">
                                                <div className="text-xs text-slate-500 uppercase mb-2">Lời thoại được nhận diện:</div>
                                                <p className="text-sm text-slate-400 italic">"{result.text_spoken}"</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </motion.section>
                        )}
                    </AnimatePresence>
                </main>

                <footer className="mt-20 text-center text-slate-600 text-sm">
                    <div className="flex items-center justify-center space-x-6 mb-4">
                        <span className="hover:text-blue-500 transition-colors cursor-help">Tài liệu</span>
                        <span className="hover:text-blue-500 transition-colors cursor-help">Cộng đồng</span>
                        <span className="hover:text-blue-500 transition-colors cursor-help">Hỗ trợ</span>
                    </div>
                    <p>© 2026 The MC Hub. Xây dựng bởi công nghệ Advanced Agentic Coding.</p>
                </footer>
            </div>
        </div>
    );
}

export default App;
