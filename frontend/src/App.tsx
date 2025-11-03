import { useState } from "react";
import { ImageUpload } from "./components/ImageUpload";
import { JokeCard } from "./components/JokeCard";
import { Button } from "./components/ui/button";
import { Sparkles, Loader2 } from "lucide-react";

interface Joke {
  id: number;
  text: string;
  funnyScore: number;
  relevanceScore: number;
}

export default function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [jokes, setJokes] = useState<Joke[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const generateJokes = async () => {
    if (!imageUrl) return;
    
    setIsGenerating(true);
    
    // Simulate joke generation
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const mockJokes = [
      "When Photoshop decided your day wasn't pretty enough, it added some filters... and your ex in the background!",
      "This photo is so epic, even the camera asked for an autograph!",
      "If this picture could talk, it would say: 'I'm too cool for this world!'"
    ];
    
    const generatedJokes = mockJokes.map((text, index) => ({
      id: Date.now() + index,
      text,
      funnyScore: 0,
      relevanceScore: 0
    }));
    
    setJokes(generatedJokes);
    setIsGenerating(false);
  };

  const handleRateFunny = (jokeId: number, rating: number) => {
    setJokes(prev => 
      prev.map(joke => 
        joke.id === jokeId ? { ...joke, funnyScore: rating } : joke
      )
    );
  };

  const handleRateRelevance = (jokeId: number, rating: number) => {
    setJokes(prev => 
      prev.map(joke => 
        joke.id === jokeId ? { ...joke, relevanceScore: rating } : joke
      )
    );
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="mb-3 flex items-center justify-center gap-3">
            <Sparkles className="w-8 h-8 text-primary" />
            Image Joke Generator
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload any image and our AI will create hilarious jokes for you!
          </p>
        </div>

        {/* Image Upload Section */}
        <div className="mb-8">
          <ImageUpload onImageSelect={setImageUrl} />
        </div>

        {/* Generate Button */}
        <div className="flex justify-center mb-12">
          <Button
            onClick={generateJokes}
            disabled={!imageUrl || isGenerating}
            size="lg"
            className="min-w-[200px]"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Generate Jokes
              </>
            )}
          </Button>
        </div>

        {/* Jokes Grid */}
        {jokes.length > 0 && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h2 className="mb-6 text-center">Generated Jokes</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {jokes.map((joke) => (
                <JokeCard
                  key={joke.id}
                  joke={joke.text}
                  funnyScore={joke.funnyScore}
                  relevanceScore={joke.relevanceScore}
                  onRateFunny={(rating) => handleRateFunny(joke.id, rating)}
                  onRateRelevance={(rating) => handleRateRelevance(joke.id, rating)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!imageUrl && jokes.length === 0 && (
          <div className="text-center text-muted-foreground py-12">
            <p>Upload an image to start generating jokes</p>
          </div>
        )}
      </div>
    </div>
  );
}