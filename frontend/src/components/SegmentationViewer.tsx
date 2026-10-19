import { useState, useRef, useCallback, useEffect } from 'react';
import { Eye, EyeOff, ChevronsLeftRight } from 'lucide-react';
import type { InferenceResult } from '../constants';

interface Props {
  originalSrc: string;
  result: InferenceResult;
}

export default function SegmentationViewer({ originalSrc, result }: Props) {
  const [showMask, setShowMask] = useState(true);
  const [sliderPct, setSliderPct] = useState(50);
  const [mode, setMode] = useState<'grid' | 'compare'>('grid');
  const sliderRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const maskSrc = `data:image/png;base64,${result.mask_b64}`;
  const overlaySrc = `data:image/png;base64,${result.overlay_b64}`;

  // Comparison slider drag
  const onPointerMove = useCallback((e: PointerEvent) => {
    if (!dragging.current || !sliderRef.current) return;
    const rect = sliderRef.current.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    setSliderPct(pct);
  }, []);

  const onPointerUp = useCallback(() => { dragging.current = false; }, []);

  useEffect(() => {
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, [onPointerMove, onPointerUp]);

  return (
    <div>
      {/* Controls row */}
      <div className="flex gap-8 flex-wrap" style={{ marginBottom: 16 }}>
        <button
          id="toggle-mask-btn"
          className={`toggle-btn${showMask ? ' active' : ''}`}
          onClick={() => setShowMask(v => !v)}
          aria-pressed={showMask}
        >
          {showMask ? <Eye size={13} /> : <EyeOff size={13} />}
          {showMask ? 'Mask On' : 'Mask Off'}
        </button>
        <button
          id="grid-mode-btn"
          className={`toggle-btn${mode === 'grid' ? ' active' : ''}`}
          onClick={() => setMode('grid')}
          aria-pressed={mode === 'grid'}
        >
          Grid View
        </button>
        <button
          id="compare-mode-btn"
          className={`toggle-btn${mode === 'compare' ? ' active' : ''}`}
          onClick={() => setMode('compare')}
          aria-pressed={mode === 'compare'}
        >
          <ChevronsLeftRight size={13} /> Compare
        </button>
      </div>

      {/* Grid view */}
      {mode === 'grid' && (
        <div className="seg-viewer">
          {/* Original */}
          <div className="seg-panel">
            <div className="seg-panel__header">
              <span>Original Image</span>
              <span className="label label-ghost" style={{ color: 'var(--ivory)', borderColor: 'rgba(255,255,255,0.3)' }}>
                {result.input_w}×{result.input_h}
              </span>
            </div>
            <img
              src={originalSrc}
              alt="Original dermoscopic image"
              className="seg-panel__img"
            />
          </div>

          {/* Mask */}
          {showMask && (
            <div className="seg-panel">
              <div className="seg-panel__header">
                <span>Predicted Mask</span>
                <span className="label label-yellow">Binary</span>
              </div>
              <img
                src={maskSrc}
                alt="Predicted binary segmentation mask"
                className="seg-panel__img"
              />
            </div>
          )}

          {/* Overlay */}
          <div className="seg-panel" style={!showMask ? { gridColumn: '2' } : {}}>
            <div className="seg-panel__header">
              <span>Overlay</span>
              <span className="label label-coral">α=0.42</span>
            </div>
            <img
              src={overlaySrc}
              alt="Segmentation overlay on original image"
              className="seg-panel__img"
            />
          </div>
        </div>
      )}

      {/* Comparison slider */}
      {mode === 'compare' && (
        <div
          id="compare-slider"
          ref={sliderRef}
          className="compare-slider"
          onPointerDown={() => { dragging.current = true; }}
          style={{ maxWidth: 640, margin: '0 auto' }}
          aria-label="Image comparison slider: drag to compare original and overlay"
        >
          {/* Base: original */}
          <img
            src={originalSrc}
            alt="Original"
            className="compare-slider__img"
            draggable={false}
          />
          {/* Top: overlay */}
          <div
            className="compare-slider__overlay"
            style={{ width: `${sliderPct}%` }}
          >
            <img
              src={overlaySrc}
              alt="Overlay"
              style={{ width: sliderRef.current?.offsetWidth ?? 640, display: 'block' }}
              draggable={false}
            />
          </div>
          {/* Divider + handle */}
          <div className="compare-slider__divider" style={{ left: `${sliderPct}%` }}>
            <div className="compare-slider__handle">
              <ChevronsLeftRight size={16} />
            </div>
          </div>
          {/* Labels */}
          <div className="compare-slider__labels">
            <span className="compare-label">Overlay</span>
            <span className="compare-label">Original</span>
          </div>
        </div>
      )}
    </div>
  );
}
