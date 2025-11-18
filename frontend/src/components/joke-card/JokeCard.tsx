import StarRating from '../star-rating/StarRating';
import './joke-card.css';

interface JokeCardProps {
  joke: string;
  funnyScore: number;
  relevanceScore: number;
  onFunnyRatingChange: (score: number) => void;
  onRelevanceRatingChange: (score: number) => void;
}

export default function JokeCard({
  joke,
  funnyScore,
  relevanceScore,
  onFunnyRatingChange,
  onRelevanceRatingChange,
}: JokeCardProps) {
  return (
    <div className="joke-card">
      <div className="joke-card__content">
        <p className="joke-card__text">{joke}</p>
      </div>
      
      <div className="joke-card__ratings">
        <div className="joke-card__rating-item">
          <label className="joke-card__rating-label">
            Funny Score
          </label>
          <StarRating
            score={funnyScore}
            onScoreChange={onFunnyRatingChange}
          />
        </div>
        
        <div className="joke-card__rating-item">
          <label className="joke-card__rating-label">
            Relevance Score
          </label>
          <StarRating
            score={relevanceScore}
            onScoreChange={onRelevanceRatingChange}
          />
        </div>
      </div>
    </div>
  );
}
