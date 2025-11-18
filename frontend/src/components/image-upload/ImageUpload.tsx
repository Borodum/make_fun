import { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';
import './image-upload.css';

interface ImageUploadProps {
  onImageUpload: (imageUrl: string) => void;
}

export default function ImageUpload({ onImageUpload }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = e => {
          if (e.target?.result) {
            onImageUpload(e.target.result as string);
          }
        };
        reader.readAsDataURL(file);
      }
    },
    [onImageUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      
      const file = e.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData.items;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (file) {
            handleFile(file);
          }
          break;
        }
      }
    },
    [handleFile]
  );

  return (
    <div
      className={`image-upload ${isDragging ? 'image-upload--dragging' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onPaste={handlePaste}
      tabIndex={0}
    >
      <div className="image-upload__icon-container">
        {isDragging ? (
          <ImageIcon className="image-upload__icon" />
        ) : (
          <Upload className="image-upload__icon" />
        )}
      </div>
      
      <div className="image-upload__text">
        <p className="image-upload__title">
          {isDragging ? 'Drop your image here' : 'Upload an image'}
        </p>
        <p className="image-upload__subtitle">
          Drag and drop, paste, or click to select
        </p>
      </div>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="image-upload__input"
        id="file-upload"
      />
      
      <label htmlFor="file-upload" className="image-upload__button">
        Choose File
      </label>
    </div>
  );
}
