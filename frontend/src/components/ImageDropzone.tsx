import { useRef, useState, useCallback } from 'react';
import { Upload, ImageIcon } from 'lucide-react';

interface Props {
  onFile: (file: File) => void;
}

const ACCEPTED = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
const MAX_BYTES = 20 * 1024 * 1024; // 20 MB

export default function ImageDropzone({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState('');

  const validate = useCallback((file: File): boolean => {
    setError('');
    if (!ACCEPTED.includes(file.type)) {
      setError(`Unsupported format. Use PNG, JPG, or BMP.`);
      return false;
    }
    if (file.size > MAX_BYTES) {
      setError(`File too large (max 20 MB). Yours: ${(file.size / 1024 / 1024).toFixed(1)} MB.`);
      return false;
    }
    return true;
  }, []);

  const handleFile = useCallback((file: File) => {
    if (validate(file)) onFile(file);
  }, [validate, onFile]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  };

  return (
    <div>
      <div
        id="image-dropzone"
        className={`dropzone${dragOver ? ' drag-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload dermoscopic image for segmentation"
        onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          className="dropzone__input"
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/bmp"
          onChange={onInputChange}
          aria-hidden="true"
          tabIndex={-1}
          id="file-upload-input"
        />
        <ImageIcon
          className="dropzone__icon"
          size={42}
          strokeWidth={1.2}
          aria-hidden="true"
        />
        <div className="dropzone__title">
          {dragOver ? 'Release to upload' : 'Drop a dermoscopic image'}
        </div>
        <p className="dropzone__sub">
          or click to browse your files
        </p>
        <button
          type="button"
          className="btn btn-teal btn-sm"
          onClick={e => { e.stopPropagation(); inputRef.current?.click(); }}
          id="browse-btn"
        >
          <Upload size={14} /> Browse File
        </button>
        <p className="dropzone__formats">
          PNG · JPG · BMP — max 20 MB
        </p>
      </div>
      {error && (
        <p
          role="alert"
          style={{
            color: 'var(--vermilion)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            marginTop: 8,
          }}
        >
          ⚠ {error}
        </p>
      )}
    </div>
  );
}
