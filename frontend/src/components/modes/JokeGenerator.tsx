import { useState } from 'react';
import ImageUpload from '../image-upload/ImageUpload';
import JokeCard from '../joke-card/JokeCard';

interface Joke {
  id: number;
  text: string;
  funnyScore: number;
  relevanceScore: number;
}

export default function JokeGenerator() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [jokes, setJokes] = useState<Joke[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleImageUpload = (imageUrl: string) => {
    setUploadedImage(imageUrl);
    setJokes([]);
  };

  const generateJokes = async () => {
    setIsGenerating(true);

    let apiJokes: string[] | null = null;

    try {
      // If there is an uploaded image, send it to the backend which returns jokes
      if (uploadedImage) {
        const res = await fetch("http://127.0.0.1:8000/images/upload/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ url: uploadedImage }),
        });
        const uploadData = await res.json();
        if (uploadData && Array.isArray(uploadData.jokes)) {
          apiJokes = uploadData.jokes;
        }
      }
    } catch (e) {
      console.error('Error while uploading image or requesting jokes:', e);
    }

    // If API didn't return jokes, use the existing mock jokes
    if (!apiJokes) {
      apiJokes = [
        "When you realize your houseplant has better living conditions than you do.",
        "This photo screams 'I'm not like other plants, I have a personality.'",
        "Plot twist: The plant is actually running the household.",
      ];
    }

    // Map API or mock jokes into the Joke type and set state
    const mapped: Joke[] = apiJokes.map((text, i) => ({
      id: i + 1,
      text,
      funnyScore: 0,
      relevanceScore: 0,
    }));

    setJokes(mapped);
    setIsGenerating(false);
  };

  const updateJokeRating = (
    jokeId: number,
    ratingType: 'funny' | 'relevance',
    score: number
  ) => {
    setJokes(prevJokes =>
      prevJokes.map(joke =>
        joke.id === jokeId
          ? {
              ...joke,
              ...(ratingType === 'funny'
                ? { funnyScore: score }
                : { relevanceScore: score }),
            }
          : joke
      )
    );
  };

  return (
    <div>
      {/* Image Upload Section */}
      <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
        <ImageUpload onImageUpload={handleImageUpload} />
        
        {uploadedImage && (
          <div className="mt-6">
            <img
              src={uploadedImage}
              alt="Uploaded"
              className="max-w-md mx-auto rounded-xl shadow-md"
            />
            <div className="text-center mt-6">
              <button
                onClick={generateJokes}
                disabled={isGenerating}
                className="bg-[#032D68] text-white px-8 py-3 rounded-full hover:bg-[#04397a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
              >
                {isGenerating ? 'Generating Jokes...' : 'Generate Jokes'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Jokes Display Section */}
      {jokes.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {jokes.map(joke => (
            <JokeCard
              key={joke.id}
              joke={joke.text}
              funnyScore={joke.funnyScore}
              relevanceScore={joke.relevanceScore}
              onFunnyRatingChange={score => updateJokeRating(joke.id, 'funny', score)}
              onRelevanceRatingChange={score => updateJokeRating(joke.id, 'relevance', score)}
            />
          ))}
        </div>
      )}
    </div>
  );
}