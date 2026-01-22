import { useState } from 'react';
import { ChatWindow } from './components/chat/ChatWindow';
import { VisualizerOverlay } from './components/visualizer/VisualizerOverlay';

function App() {
  const [isVisualizerOpen, setIsVisualizerOpen] = useState(false);

  return (
    <div className="h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-rose-50 overflow-hidden">
      {/* Main chat interface */}
      <ChatWindow onOpenVisualizer={() => setIsVisualizerOpen(true)} />
      
      {/* Visualizer overlay */}
      <VisualizerOverlay 
        isOpen={isVisualizerOpen} 
        onClose={() => setIsVisualizerOpen(false)} 
      />
    </div>
  );
}

export default App;
