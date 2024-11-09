import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import './App.css';

function App() {
  const [currentChat, setCurrentChat] = useState([]);
  const [chatSessions, setChatSessions] = useState([]);
  const [ragEnabled, setRagEnabled] = useState(true);
  const [chatId, setChatId] = useState(0);

  const handleNewChat = () => {
    if (currentChat.length > 0) {
      setChatSessions([...chatSessions, { id: chatId, messages: currentChat }]);
      setChatId(chatId + 1);
    }
    setCurrentChat([]);
  };

  const handleSendMessage = async (message) => {
    if (message.trim() === '') return;

    const userMessage = { text: message, type: 'user', timestamp: new Date().toLocaleString() };
    const updatedChat = [...currentChat, userMessage];
    setCurrentChat(updatedChat);

    try {
      const body = {
        user_query: message
      };
      const url = `http://localhost:8001/answer?isRAGEnabled=${ragEnabled}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      setCurrentChat([...updatedChat, { text: data.answer, type: 'bot', timestamp: new Date().toLocaleString() }]);
    } catch (error) {
      console.error('Error fetching data:', error);
    }

    // setTimeout(() => {
    //   const botResponse = generateBotResponse(message);
    //   setCurrentChat([...updatedChat, { text: botResponse, type: 'bot', timestamp: new Date().toLocaleString() }]);
    // }, 500);
  };

  // const generateBotResponse = (userMessage) => {
  //   return 'This is a simulated bot response.';
  // };

  const handleToggleRAG = () => {
    setRagEnabled(!ragEnabled);
  };

  const handleSelectModel = (model) => {
    console.log(`Selected Model: ${model}`);
  };

  return (
    <div className="container" style={{ fontFamily: 'Lato, sans-serif' }}>
      <Sidebar
        chatSessions={chatSessions}
        handleNewChat={handleNewChat}
        setCurrentChat={setCurrentChat}
      />
      <Chat
        currentChat={currentChat}
        onSendMessage={handleSendMessage}
        onToggleRAG={handleToggleRAG}
        ragEnabled={ragEnabled}
        onSelectModel={handleSelectModel}
      />
    </div>
  );
}

export default App;
