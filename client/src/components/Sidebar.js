import React, { useState } from 'react';
import { FaPlusCircle } from "react-icons/fa";
import { ReactComponent as BotIcon } from '../assets/bot-icon.svg';

function Sidebar({ chatSessions, handleNewChat, setCurrentChat }) {
    const [currentSession, setCurrentSession] = useState(null);

    const loadChatSession = (session) => {
        setCurrentChat(session.messages);
        setCurrentSession(session);
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-heading">
                <div className='sidebar-title-container'>
                    <span className="sidebar-title">LLM - RAG Enabled</span>
                </div>
                <div className='sidebar-logo-wrapper'>
                    <BotIcon/>
                </div>
            </div>

            <button onClick={handleNewChat} className="new-chat-button"><FaPlusCircle /> New Chat</button>
            <ul className="chat-log-list">
            {chatSessions.map((session) => (
                <li key={session.id} onClick={() => loadChatSession(session)} className={currentSession?.id === session.id ? 'active' : ''}>
                {/* {session.messages[session.messages.length - 1]?.text || `Chat ${session.id}`} */}
                <div className="chat-log-heading">
                    <span>{session.messages[0]?.text}</span>
                </div>
                <div className="chat-log-metadata">
                    <span>{`${session.messages.length} messages`}</span>
                    <span>{session.messages[session.messages.length - 1]?.timestamp}</span>
                </div>
                </li>
            ))}
            </ul>
        </aside>
    );
}

export default Sidebar;
