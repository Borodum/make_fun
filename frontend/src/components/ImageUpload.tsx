import { useRef, useState } from "react";
import { Upload, Image as ImageIcon } from "lucide-react";
import { ImageWithFallback } from "./figma/ImageWithFallback";

interface ImageUploadProps {
  onImageSelect: (imageUrl: string) => void;
}

export function ImageUpload({ onImageSelect }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setPreview(result);
        onImageSelect(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith("image/")) {
        const file = items[i].getAsFile();
        if (file) {
          handleFile(file);
        }
        break;
      }
    }
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
      />
      
      <div
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onPaste={handlePaste}
        tabIndex={0}
        className={`
          relative w-full aspect-[4/3] max-h-[500px] border-2 border-dashed rounded-lg
          cursor-pointer transition-all duration-200
          ${isDragging 
            ? "border-primary bg-primary/5 scale-[1.02]" 
            : "border-border hover:border-primary/50 hover:bg-accent/50"
          }
          ${preview ? "border-solid" : ""}
        `}
      >
        {preview ? (
          <div className="relative w-full h-full rounded-lg overflow-hidden group">
            <ImageWithFallback
              src={preview}
              alt="Uploaded preview"
              className="w-full h-full object-contain"
            />
            <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
              <div className="text-white text-center">
                <Upload className="w-8 h-8 mx-auto mb-2" />
                <p>Click to change</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8 text-center">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
              <ImageIcon className="w-8 h-8 text-primary" />
            </div>
            <div>
              <p className="mb-2">
                <span className="text-primary">Click to select</span> or drag and drop an image
              </p>
              <p className="text-sm text-muted-foreground">
                You can also paste an image from your clipboard (Ctrl+V)
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}