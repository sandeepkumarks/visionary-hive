import React, { useState } from 'react';
import { FaPaperPlane } from "react-icons/fa";
import { ReactComponent as BotIcon } from '../assets/bot-icon.svg';

function Chat({ currentChat, onSendMessage, onToggleRAG, ragEnabled, onSelectModel }) {
  const [input, setInput] = useState('');

  const handleSend = () => {
    onSendMessage(input);
    setInput('');
  };

  const handleEnterKey = (e) => {
    if (e.key === 'Enter') handleSend();
  };

  return (
    <main className="chat-container">
      <header className="chat-header">
        <span className="header-label">Conversation Notes</span>
        <span className="header-metadata">{currentChat.length} messages</span>
      </header>

      <div className="chat-window">
        <div className="chat-log">
            {currentChat.map((msg, index) => (
                <div key={index} className={`message-container ${msg.type === 'user' ? 'user-message' : 'bot-message'}`}>
                    {
                        msg.type === 'bot' &&
                        <div className="bot-message-icon-container">
                            <div className="bot-message-icon">
                                <BotIcon/>
                            </div>
                        </div>
                    }
                    <div className="message">
                        {msg.text}
                    </div>
                    <span className="message-timestamp">{msg.timestamp}</span>
                </div>
            ))}
        </div>
      </div>

      <div className="input-container">
        <div className="input-wrapper">
          <button onClick={onToggleRAG} className={`toggle-button ${ragEnabled ? 'on' : ''}`}>
            {ragEnabled ? 'RAG On' : 'RAG Off'}
          </button>
          <textarea
            className="user-input"
            rows="4"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleEnterKey}
            placeholder="Ask a question..."
          />
        </div>
        <div className="send-button-wrapper">
            <button onClick={handleSend} className="send-button"><FaPaperPlane />Send</button>
        </div>
      </div>
    </main>
  );
}

export default Chat;
