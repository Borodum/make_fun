import { Star } from 'lucide-react';
import './star-rating.css';

interface StarRatingProps {
  score: number;
  onScoreChange: (score: number) => void;
}

export default function StarRating({ score, onScoreChange }: StarRatingProps) {
  return (
    <div className="star-rating">
      {[1, 2, 3, 4, 5].map(starValue => (
        <button
          key={starValue}
          onClick={() => onScoreChange(starValue)}
          className="star-rating__button"
          aria-label={`Rate ${starValue} stars`}
        >
          <Star
            className={`star-rating__icon ${
              starValue <= score ? 'star-rating__icon--filled' : ''
            }`}
            fill={starValue <= score ? 'currentColor' : 'none'}
          />
        </button>
      ))}
    </div>
  );
}
