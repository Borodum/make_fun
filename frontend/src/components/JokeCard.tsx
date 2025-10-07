import { Card, CardContent } from "./ui/card";
import { StarRating } from "./StarRating";
import { Smile } from "lucide-react";

interface JokeCardProps {
  joke: string;
  onRate?: (rating: number) => void;
}

export function JokeCard({ joke, onRate }: JokeCardProps) {
  return (
    <Card className="h-full flex flex-col">
      <CardContent className="pt-6 flex flex-col gap-4 flex-1">
        <div className="flex items-start gap-3">
          <Smile className="w-5 h-5 text-primary shrink-0 mt-1" />
          <p className="flex-1">{joke}</p>
        </div>
        <div className="mt-auto pt-4 border-t border-border">
          <p className="text-sm text-muted-foreground mb-2">Оцените шутку:</p>
          <StarRating onRate={onRate} />
        </div>
      </CardContent>
    </Card>
  );
}