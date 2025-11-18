import { useState, useEffect } from 'react';
import ImageFrame from '../image-frame/ImageFrame';

interface ImageResult {
  id: number;
  url: string;
  description: string;
  relevanceScore: number;
}

export default function ImageSearch() {
  const [searchQuery, setSearchQuery] = useState('');
  const [images, setImages] = useState<ImageResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);

  // Allow configuring backend URL via Vite env var `VITE_BACKEND_URL`.
  // Fallback to localhost:8000 which is the default used by the project README.
  const BACKEND = (import.meta as any).env?.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

  useEffect(() => {
    // Quick health check so users see a nicer error than a console connection refused.
    (async () => {
      try {
        const r = await fetch(`${BACKEND}/health`);
        setBackendReachable(r.ok);
      } catch (e) {
        setBackendReachable(false);
      }
    })();
  }, [BACKEND]);

  const handleSearch = () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);

    // Call backend /text2img/upload/ to get image paths from Qdrant
    (async () => {
      try {
        if (backendReachable === false) {
          throw new Error(`Backend not reachable at ${BACKEND}`);
        }

        const resp = await fetch(`${BACKEND}/text2img/upload/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: searchQuery, top_k: 8 }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        // backend returns { paths: ["datasets/oxford_hic/images/..."] }
        const base = `${BACKEND}/static/`;
        const mapped: ImageResult[] = (data.paths || []).map((p: string, i: number) => {
          // remove leading 'datasets/' if present
          let rel = p;
          if (rel.startsWith('datasets/')) rel = rel.slice('datasets/'.length);
          const url = base + rel;
          const desc = rel.split('/').slice(-1)[0];
          return {
            id: i + 1,
            url,
            description: desc,
            relevanceScore: Math.round(100 - i * (100 / ((data.paths || []).length || 1))),
          };
        });
        setImages(mapped);
      } catch (e) {
        console.error('Image search failed', e);
        // fallback: clear images
        setImages([]);
      } finally {
        setIsSearching(false);
      }
    })();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div>
      {/* Search Section */}
      <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
        <div className="max-w-2xl mx-auto">
          <label className="block text-gray-700 mb-3">
            Describe what you're looking for:
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="E.g., cute cat, sunset beach, mountain landscape..."
              className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-xl focus:border-[#032D68] focus:outline-none transition-colors"
            />
            <button
              onClick={handleSearch}
              disabled={isSearching || !searchQuery.trim()}
              className="bg-[#032D68] text-white px-8 py-3 rounded-xl hover:bg-[#04397a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </div>
          <p className="text-gray-500 text-sm mt-3">
            Enter a description and we'll find the most relevant images from our dataset
          </p>
          {backendReachable === null && (
            <p className="text-gray-500 text-sm mt-2">Checking backend availability...</p>
          )}
          {backendReachable === false && (
            <p className="text-red-500 text-sm mt-2">Backend not reachable at {BACKEND}. Start the backend (see backend/README.md) or set `VITE_BACKEND_URL` to the correct address.</p>
          )}
        </div>
      </div>

      {/* Results Section */}
      {images.length > 0 && (
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-gray-800 mb-6">
            Found {images.length} matching images
          </h2>
          
          {/* Image Grid - 4 columns, 2 rows */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {images.map(image => (
              <ImageFrame
                key={image.id}
                imageUrl={image.url}
                description={image.description}
                relevanceScore={image.relevanceScore}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}