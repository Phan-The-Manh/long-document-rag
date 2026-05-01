import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [docLink, setDocLink] = useState('');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const sessionId = useRef(crypto.randomUUID());
  const ingestedDocUrl = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, status]);

  const handleSubmit = async () => {
    if (!docLink.trim() || !question.trim() || isLoading) return;

    setMessages(prev => [...prev, { sender: 'user', text: question }]);
    setIsLoading(true);
    setStatus('');
    const currentQuestion = question;
    setQuestion('');

    const needsIngest = docLink.trim() !== ingestedDocUrl.current;
    const endpoint = needsIngest ? '/ingest-and-chat' : '/chat';
    const body = needsIngest
      ? { doc_link: docLink, source_name: 'user_upload', message: currentQuestion, session_id: sessionId.current }
      : { message: currentQuestion, session_id: sessionId.current };

    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json();
        setMessages(prev => [...prev, { sender: 'bot', text: `Error: ${err.detail}` }]);
        setIsLoading(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let botText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const raw = decoder.decode(value);
        for (const line of raw.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch { continue; }

          if (evt.type === 'progress') {
            setStatus(evt.msg);
          } else if (evt.type === 'token') {
            botText += evt.msg;
            const snapshot = botText;
            setMessages(prev => [
              ...prev.filter(m => m.sender !== 'bot-pending'),
              { sender: 'bot-pending', text: snapshot },
            ]);
          } else if (evt.type === 'done') {
            if (needsIngest) ingestedDocUrl.current = docLink.trim();
            setMessages(prev =>
              prev.map(m => m.sender === 'bot-pending' ? { ...m, sender: 'bot' } : m)
            );
            setStatus('');
            setIsLoading(false);
          } else if (evt.type === 'error') {
            setMessages(prev => [
              ...prev.filter(m => m.sender !== 'bot-pending'),
              { sender: 'bot', text: `Error: ${evt.msg}` },
            ]);
            setStatus('');
            setIsLoading(false);
          }
        }
      }
    } catch (e) {
      setMessages(prev => [...prev, { sender: 'bot', text: `Network error: ${e.message}` }]);
      setStatus('');
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 className="chatbot-title">DocRAG Assistant</h2>

      {/* Chat history */}
      <div style={styles.chatBox}>
        {messages.length === 0 && (
          <p style={styles.placeholder}>Enter a document URL and question below to get started.</p>
        )}
        {messages.map((msg, idx) => {
          const isBot = msg.sender === 'bot' || msg.sender === 'bot-pending';
          const isUser = msg.sender === 'user';
          const [mainContent, sourcesContent] = isBot && msg.text.includes('Sources List:')
            ? msg.text.split(/\*{0,2}\s*Sources List:\s*\*{0,2}/i)
            : [msg.text, null];

          return (
            <div
              key={idx}
              style={{
                display: 'flex',
                flexDirection: isUser ? 'row-reverse' : 'row',
                alignItems: 'flex-end',
                gap: '0.5rem',
                alignSelf: isUser ? 'flex-end' : 'flex-start',
                maxWidth: '85%',
              }}
            >
              <div style={isUser ? styles.userAvatar : styles.botAvatar}>
                {isUser ? 'U' : 'AI'}
              </div>
              <div
                style={{
                  ...styles.message,
                  background: isUser ? '#0084ff' : '#f0f0f0',
                  color: isUser ? '#fff' : '#222',
                }}
              >
                {isBot ? (
                  <>
                    <div className="bot-md"><ReactMarkdown>{mainContent}</ReactMarkdown></div>
                    {sourcesContent && (
                      <>
                        <hr style={styles.divider} />
                        <div style={styles.sourcesLabel}>Sources</div>
                        <div style={{ fontSize: '0.88rem', color: '#555', lineHeight: '1.6' }}>
                          {sourcesContent.trim().split('\n').filter(l => l.trim()).map((line, i) => (
                            <div key={i}>{line.trim().replace(/^-\s*/, '• ')}</div>
                          ))}
                        </div>
                      </>
                    )}
                    {msg.sender === 'bot-pending' && <span style={styles.cursor}>▌</span>}
                  </>
                ) : (
                  msg.text
                )}
              </div>
            </div>
          );
        })}
        {isLoading && !messages.some(m => m.sender === 'bot-pending') && !status && (
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.5rem', alignSelf: 'flex-start' }}>
            <div style={styles.botAvatar}>AI</div>
            <div style={{ ...styles.message, background: '#f0f0f0' }}>
              <div className="typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        {status && (
          <div style={styles.statusBubble}>{status}</div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Inputs */}
      <div style={styles.inputArea}>
        <input
          type="text"
          value={docLink}
          onChange={e => setDocLink(e.target.value)}
          placeholder="Document URL (PDF link)..."
          disabled={isLoading}
          style={styles.input}
        />
        <div style={styles.questionRow}>
          <input
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            placeholder="Ask a question..."
            disabled={isLoading}
            style={{ ...styles.input, flex: 1 }}
          />
          <button
            onClick={handleSubmit}
            disabled={isLoading || !docLink.trim() || !question.trim()}
            style={{
              ...styles.button,
              opacity: isLoading || !docLink.trim() || !question.trim() ? 0.5 : 1,
            }}
          >
            {isLoading ? '...' : 'Ask'}
          </button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '680px',
    margin: '2rem auto',
    padding: '0 1rem',
    fontFamily: 'sans-serif',
    display: 'flex',
    flexDirection: 'column',
    height: '90vh',
  },
  botAvatar: {
    width: '34px',
    height: '34px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #0084ff 0%, #7c3aed 100%)',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '0.7rem',
    fontWeight: 'bold',
    flexShrink: 0,
    boxShadow: '0 1px 4px rgba(0,132,255,0.35)',
  },
  userAvatar: {
    width: '34px',
    height: '34px',
    borderRadius: '50%',
    background: '#444',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '0.7rem',
    fontWeight: 'bold',
    flexShrink: 0,
  },
  chatBox: {
    flex: 1,
    border: '1px solid #ddd',
    borderRadius: '8px',
    padding: '1rem',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    background: '#fafafa',
    marginBottom: '0.75rem',
  },
  placeholder: {
    color: '#aaa',
    textAlign: 'center',
    marginTop: '2rem',
    fontSize: '0.9rem',
  },
  message: {
    padding: '0.6rem 0.9rem',
    borderRadius: '16px',
    lineHeight: '1.5',

    wordBreak: 'break-word',
    fontSize: '0.95rem',
    flex: 1,
  },
  cursor: {
    animation: 'blink 1s step-start infinite',
    marginLeft: '2px',
  },
  statusBubble: {
    alignSelf: 'center',
    background: '#e8f0fe',
    color: '#1a73e8',
    borderRadius: '12px',
    padding: '0.4rem 0.9rem',
    fontSize: '0.85rem',
    fontStyle: 'italic',
  },
  inputArea: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  questionRow: {
    display: 'flex',
    gap: '0.5rem',
  },
  input: {
    padding: '0.55rem 0.75rem',
    borderRadius: '8px',
    border: '1px solid #ccc',
    fontSize: '0.95rem',
    outline: 'none',
    width: '100%',
    boxSizing: 'border-box',
  },
  divider: {
    border: 'none',
    borderTop: '1px solid #ccc',
    margin: '0.6rem 0',
  },
  sourcesLabel: {
    fontSize: '0.78rem',
    fontWeight: 'bold',
    color: '#666',
    marginBottom: '0.2rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  button: {
    padding: '0.55rem 1.2rem',
    borderRadius: '8px',
    border: 'none',
    background: '#0084ff',
    color: '#fff',
    fontWeight: 'bold',
    cursor: 'pointer',
    fontSize: '0.95rem',
    whiteSpace: 'nowrap',
  },
};

export default App;
