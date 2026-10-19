import { useState, useRef, useCallback } from 'react';
import { RefreshCw, AlertTriangle, CheckCircle, Zap } from 'lucide-react';
import ImageDropzone from '../components/ImageDropzone';
import ImagePreviewPanel from '../components/ImagePreviewPanel';
import SegmentationViewer from '../components/SegmentationViewer';
import ResultsPanel from '../components/ResultsPanel';
import { API_BASE, type InferenceResult, type AppStatus } from '../constants';

interface ImageMeta {
  file: File;
  objectUrl: string;
  width: number;
  height: number;
}

function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => { resolve({ width: img.naturalWidth, height: img.naturalHeight }); URL.revokeObjectURL(url); };
    img.onerror = reject;
    img.src = url;
  });
}

export default function AnalyzePage() {
  const [imageMeta, setImageMeta] = useState<ImageMeta | null>(null);
  const [status, setStatus] = useState<AppStatus>('idle');
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    if (imageMeta) URL.revokeObjectURL(imageMeta.objectUrl);
    setImageMeta(null);
    setStatus('idle');
    setResult(null);
    setErrorMsg('');
  }, [imageMeta]);

  const handleFile = useCallback(async (file: File) => {
    if (imageMeta) URL.revokeObjectURL(imageMeta.objectUrl);
    const { width, height } = await getImageDimensions(file);
    const objectUrl = URL.createObjectURL(file);
    setImageMeta({ file, objectUrl, width, height });
    setStatus('idle');
    setResult(null);
    setErrorMsg('');
  }, [imageMeta]);

  const handleReplace = () => fileInputRef.current?.click();

  const runInference = async () => {
    if (!imageMeta) return;
    setStatus('loading');
    setResult(null);
    setErrorMsg('');

    const formData = new FormData();
    formData.append('file', imageMeta.file);

    try {
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      const data: InferenceResult = await res.json();
      setResult(data);
      setStatus('success');
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error');
      setStatus('error');
    }
  };

  return (
    <main className="page-content">
      {/* Hidden file input for replace */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/bmp"
        style={{ display: 'none' }}
        id="replace-file-input"
        onChange={async e => {
          const f = e.target.files?.[0];
          if (f) await handleFile(f);
          e.target.value = '';
        }}
      />

      <div className="container" style={{ paddingTop: 48, paddingBottom: 80 }}>
        {/* Page header */}
        <div style={{ marginBottom: 40 }}>
          <div className="section-eyebrow">VISTA-XL · Real Inference</div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3.2rem)', marginBottom: 12 }}>
            Image Analysis Workspace
          </h1>
          <p style={{ maxWidth: 580, opacity: 0.75 }}>
            Upload a dermoscopic image. VISTA-XL will run real inference using the trained checkpoint
            and return the actual predicted binary segmentation mask and overlay.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: imageMeta ? '1fr 1fr' : '1fr', gap: 32, maxWidth: imageMeta ? '100%' : 640 }}>
          {/* ── Left column: upload + preview ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

            {!imageMeta ? (
              <ImageDropzone onFile={handleFile} />
            ) : (
              <>
                <ImagePreviewPanel
                  src={imageMeta.objectUrl}
                  fileName={imageMeta.file.name}
                  width={imageMeta.width}
                  height={imageMeta.height}
                  onRemove={reset}
                  onReplace={handleReplace}
                />

                {/* Analyze button */}
                {status !== 'success' && (
                  <button
                    id="analyze-btn"
                    className="btn btn-primary btn-lg w-full"
                    onClick={runInference}
                    disabled={status === 'loading'}
                    style={{ justifyContent: 'center' }}
                  >
                    {status === 'loading' ? (
                      <>
                        <span className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                        Running VISTA-XL…
                      </>
                    ) : (
                      <>
                        <Zap size={18} /> Analyze Image
                      </>
                    )}
                  </button>
                )}

                {/* Reset */}
                {status === 'success' && (
                  <button
                    id="new-analysis-btn"
                    className="btn btn-secondary w-full"
                    onClick={reset}
                    style={{ justifyContent: 'center' }}
                  >
                    <RefreshCw size={16} /> New Analysis
                  </button>
                )}

                {/* Error */}
                {status === 'error' && (
                  <div className="error-box" role="alert">
                    <AlertTriangle className="error-box__icon" size={20} />
                    <div>
                      <div className="error-box__title">Inference Failed</div>
                      <p className="error-box__msg">{errorMsg}</p>
                      <p className="error-box__msg" style={{ marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: '0.7rem', opacity: 0.6 }}>
                        Make sure the FastAPI backend is running on port 8000.
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Loading state */}
            {status === 'loading' && (
              <div className="loading-state">
                <div className="loading-spinner" />
                <div className="loading-state__title">Running inference…</div>
                <p className="loading-state__sub">CPU inference may take 2–10 seconds</p>
              </div>
            )}
          </div>

          {/* ── Right column: results ── */}
          {imageMeta && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {status === 'idle' && (
                <div style={{
                  border: '2px dashed rgba(61,95,88,0.3)',
                  borderRadius: 'var(--radius)',
                  padding: '48px 24px',
                  textAlign: 'center',
                  color: 'rgba(16,32,29,0.4)',
                }}>
                  <CheckCircle size={32} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
                  <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                    Press <strong>Analyze Image</strong> to run segmentation
                  </p>
                </div>
              )}

              {status === 'success' && result && (
                <>
                  {/* Success badge */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span className="status-badge status-badge--success">Segmentation Complete</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', opacity: 0.55 }}>
                      {result.inference_ms.toFixed(0)} ms
                    </span>
                  </div>
                  {/* Visualizations */}
                  <SegmentationViewer originalSrc={imageMeta.objectUrl} result={result} />
                  {/* Metrics */}
                  <ResultsPanel result={result} />
                </>
              )}
            </div>
          )}
        </div>

        {/* Research disclaimer */}
        <div style={{
          marginTop: 48,
          padding: '16px 20px',
          background: 'var(--yellow)',
          border: 'var(--border)',
          borderRadius: 'var(--radius)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.72rem',
          maxWidth: 700,
        }}>
          ⚠ Research prototype only. Predictions are not a diagnosis and must not be used as a substitute
          for professional medical assessment.
        </div>
      </div>
    </main>
  );
}
