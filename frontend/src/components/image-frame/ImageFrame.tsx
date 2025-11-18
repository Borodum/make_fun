import './image-frame.css';

interface ImageFrameProps {
  imageUrl: string;
  description: string;
  relevanceScore: number;
}

export default function ImageFrame({ imageUrl, description, relevanceScore }: ImageFrameProps) {
  return (
    <div className="image-frame">
      <div className="image-frame__container">
        <img
          src={imageUrl}
          alt={description}
          className="image-frame__image"
        />
        <div className="image-frame__overlay">
          <div className="image-frame__score">
            {relevanceScore}% match
          </div>
        </div>
      </div>
      <div className="image-frame__description">
        {description}
      </div>
    </div>
  );
}
