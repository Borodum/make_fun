import { Card, CardContent } from "./ui/card";
import { StarRating } from "./StarRating";
import { Smile } from "lucide-react";

interface JokeCardProps {
  joke: string;
  funnyScore?: number;
  relevanceScore?: number;
  onRateFunny?: (rating: number) => void;
  onRateRelevance?: (rating: number) => void;
}

export function JokeCard({ 
  joke, 
  funnyScore = 0,
  relevanceScore = 0,
  onRateFunny, 
  onRateRelevance 
}: JokeCardProps) {
  return (
    <Card className="h-full flex flex-col">
      <CardContent className="pt-6 flex flex-col gap-4 flex-1">
        <div className="flex items-start gap-3">
          <Smile className="w-5 h-5 text-primary shrink-0 mt-1" />
          <p className="flex-1">{joke}</p>
        </div>
        <div className="mt-auto pt-4 border-t border-border space-y-3">
          <div>
            <p className="text-sm text-muted-foreground mb-2">Funny Score:</p>
            <StarRating 
              initialRating={funnyScore}
              onRate={onRateFunny} 
            />
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-2">Relevance Score:</p>
            <StarRating 
              initialRating={relevanceScore}
              onRate={onRateRelevance} 
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}