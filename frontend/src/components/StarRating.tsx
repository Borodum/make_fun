import { Star } from "lucide-react";
import { useState } from "react";

interface StarRatingProps {
  onRate?: (rating: number) => void;
  initialRating?: number;
}

export function StarRating({ onRate, initialRating = 0 }: StarRatingProps) {
  const [rating, setRating] = useState(initialRating);
  const [hover, setHover] = useState(0);

  const handleClick = (value: number) => {
    setRating(value);
    onRate?.(value);
  };

  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          onClick={() => handleClick(star)}
          onMouseEnter={() => setHover(star)}
          onMouseLeave={() => setHover(0)}
          className="transition-transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-primary rounded"
        >
          <Star
            className={`w-6 h-6 transition-colors ${
              star <= (hover || rating)
                ? "fill-yellow-400 stroke-yellow-400"
                : "fill-none stroke-muted-foreground"
            }`}
          />
        </button>
      ))}
    </div>
  );
}