import { Trash2, RefreshCw } from 'lucide-react';

interface Props {
  src: string;
  fileName: string;
  width: number;
  height: number;
  onRemove: () => void;
  onReplace: () => void;
}

export default function ImagePreviewPanel({ src, fileName, width, height, onRemove, onReplace }: Props) {
  return (
    <div className="preview-panel" role="img" aria-label={`Preview of ${fileName}`}>
      <div className="preview-panel__header">
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', opacity: 0.8 }}>
          {fileName}
        </span>
        <div className="flex gap-8">
          <button
            id="replace-image-btn"
            className="btn btn-ghost btn-sm"
            onClick={onReplace}
            aria-label="Replace image"
            style={{ color: 'var(--ivory)', borderColor: 'rgba(242,242,235,0.3)', padding: '3px 10px' }}
          >
            <RefreshCw size={12} /> Replace
          </button>
          <button
            id="remove-image-btn"
            className="btn btn-sm"
            onClick={onRemove}
            aria-label="Remove image"
            style={{
              background: 'var(--vermilion)',
              color: 'var(--ivory)',
              borderColor: 'var(--shadow)',
              padding: '3px 10px',
              boxShadow: 'none',
            }}
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>
      <img
        src={src}
        alt={`Uploaded image: ${fileName}`}
        className="preview-panel__img"
      />
      <div className="preview-panel__footer">
        Original dimensions: {width} × {height} px · Model input: 224 × 224 px
      </div>
    </div>
  );
}
