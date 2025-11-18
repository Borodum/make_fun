import { useState } from 'react';
import JokeGenerator from './components/modes/JokeGenerator';
import ImageSearch from './components/modes/ImageSearch';

export default function App() {
  const [activeMode, setActiveMode] = useState<'jokes' | 'images'>('jokes');

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-slate-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-[#032D68] mb-2">Balatro Jokers</h1>
          <p className="text-gray-600">Upload an image to generate jokes or describe to find matching images</p>
        </header>

        {/* Mode Switcher */}
        <div className="flex justify-center mb-10">
          <div className="bg-white rounded-full p-1.5 shadow-md inline-flex gap-1">
            <button
              onClick={() => setActiveMode('jokes')}
              className={`px-6 py-3 rounded-full transition-all ${
                activeMode === 'jokes'
                  ? 'bg-[#032D68] text-white shadow-md'
                  : 'text-gray-600 hover:text-[#032D68]'
              }`}
            >
              Generate Jokes
            </button>
            <button
              onClick={() => setActiveMode('images')}
              className={`px-6 py-3 rounded-full transition-all ${
                activeMode === 'images'
                  ? 'bg-[#032D68] text-white shadow-md'
                  : 'text-gray-600 hover:text-[#032D68]'
              }`}
            >
              Find Images
            </button>
          </div>
        </div>

        {/* Mode Content */}
        <div className="transition-all">
          {activeMode === 'jokes' ? <JokeGenerator /> : <ImageSearch />}
        </div>
      </div>
    </div>
  );
}