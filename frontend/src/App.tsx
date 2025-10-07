import { useState } from "react";
import { ImageUpload } from "./components/ImageUpload";
import { JokeCard } from "./components/JokeCard";
import { Button } from "./components/ui/button";
import { Sparkles, Loader2 } from "lucide-react";

interface Joke {
  id: number;
  text: string;
  rating: number;
}

export default function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [jokes, setJokes] = useState<Joke[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const generateJokes = async () => {
    if (!imageUrl) return;
    
    setIsGenerating(true);
    
    // Симуляция генерации шуток
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const mockJokes = [
      "Когда фотошоп решил, что у тебя недостаточно красивый день, он добавил немного фильтров... и твоего бывшего на задний план!",
      "Это фото настолько эпичное, что даже камера попросила автограф!",
      "Если бы эта картинка могла говорить, она бы сказала: 'Я слишком крута для этого мира!'"
    ];
    
    const generatedJokes = mockJokes.map((text, index) => ({
      id: Date.now() + index,
      text,
      rating: 0
    }));
    
    setJokes(generatedJokes);
    setIsGenerating(false);
  };

  const handleRateJoke = (jokeId: number, rating: number) => {
    setJokes(prev => 
      prev.map(joke => 
        joke.id === jokeId ? { ...joke, rating } : joke
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
            Генератор шуток из изображений
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Загрузите любое изображение, и наш ИИ создаст для вас смешные шутки!
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
                Генерация...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Сгенерировать шутки
              </>
            )}
          </Button>
        </div>

        {/* Jokes Grid */}
        {jokes.length > 0 && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h2 className="mb-6 text-center">Сгенерированные шутки</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {jokes.map((joke) => (
                <JokeCard
                  key={joke.id}
                  joke={joke.text}
                  onRate={(rating) => handleRateJoke(joke.id, rating)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!imageUrl && jokes.length === 0 && (
          <div className="text-center text-muted-foreground py-12">
            <p>Загрузите изображение, чтобы начать генерировать шутки</p>
          </div>
        )}
      </div>
    </div>
  );
}