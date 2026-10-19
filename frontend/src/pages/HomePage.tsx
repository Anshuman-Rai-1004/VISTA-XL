import { useNavigate } from 'react-router-dom';
import { ArrowRight, FlaskConical, Layers, BarChart2 } from 'lucide-react';
import { VALIDATION_METRICS, MODEL_INFO } from '../constants';

// Hero visual panel — illustrative graphic using CSS
function HeroVisual() {
  return (
    <div className="visual-panel" aria-hidden="true">
      <div style={{
        background: 'var(--ink)',
        border: 'var(--border)',
        borderRadius: 'var(--radius)',
        padding: 16,
        boxShadow: 'var(--shadow-hard)',
      }}>
        {/* Three panels */}
        <div className="visual-panel__row">
          {/* Original panel */}
          <div className="visual-panel__cell" style={{ aspectRatio: '1', background: '#1a2e2a' }}>
            <svg viewBox="0 0 100 100" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
              {/* Stylised dermoscopy illustration */}
              <circle cx="50" cy="50" r="46" fill="#2a4038" />
              <ellipse cx="50" cy="50" rx="30" ry="26" fill="#3a5a52" opacity="0.8" />
              <ellipse cx="44" cy="46" rx="18" ry="15" fill="#5a8070" opacity="0.6" />
              <ellipse cx="52" cy="52" rx="10" ry="9" fill="#7a9f94" opacity="0.5" />
              <circle cx="38" cy="42" r="4" fill="#2a1a18" opacity="0.7" />
              <circle cx="58" cy="46" r="3" fill="#2a1a18" opacity="0.5" />
              <circle cx="46" cy="58" r="5" fill="#140e0d" opacity="0.6" />
            </svg>
            <div className="visual-panel__caption">Original</div>
          </div>
          {/* Mask panel */}
          <div className="visual-panel__cell" style={{ aspectRatio: '1', background: '#000' }}>
            <svg viewBox="0 0 100 100" width="100%" height="100%">
              <ellipse cx="50" cy="50" rx="30" ry="26" fill="#fff" />
            </svg>
            <div className="visual-panel__caption">Mask</div>
          </div>
          {/* Overlay panel */}
          <div className="visual-panel__cell" style={{ aspectRatio: '1', background: '#1a2e2a' }}>
            <svg viewBox="0 0 100 100" width="100%" height="100%">
              <circle cx="50" cy="50" r="46" fill="#2a4038" />
              <ellipse cx="50" cy="50" rx="30" ry="26" fill="#3a5a52" opacity="0.8" />
              <ellipse cx="44" cy="46" rx="18" ry="15" fill="#5a8070" opacity="0.6" />
              {/* Coral overlay */}
              <ellipse cx="50" cy="50" rx="30" ry="26" fill="#E97B77" opacity="0.42" />
              <ellipse cx="50" cy="50" rx="30" ry="26" fill="none" stroke="#E53927" strokeWidth="1.5" />
            </svg>
            <div className="visual-panel__caption">Overlay</div>
          </div>
        </div>
        {/* Labels row */}
        <div style={{
          display: 'flex',
          gap: 8,
          marginTop: 12,
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}>
          <span className="label label-teal">Swin-Tiny Encoder</span>
          <span className="label label-yellow">mLSTM</span>
          <span className="label label-coral">Sky-Net Decoder</span>
        </div>
        <p style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.6rem',
          color: 'rgba(242,242,235,0.4)',
          textAlign: 'center',
          marginTop: 10,
          letterSpacing: '0.08em',
        }}>
          ILLUSTRATIVE — not a real model prediction
        </p>
      </div>
    </div>
  );
}

function MetricHighlight({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="metric-card">
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">{value}</div>
      <div className="metric-card__sub">{sub}</div>
    </div>
  );
}

export default function HomePage() {
  const navigate = useNavigate();

  const featureCards = [
    {
      icon: <FlaskConical size={20} />, tag: 'Research',
      title: 'Real Inference Engine',
      body: 'Upload any dermoscopic image and receive an actual predicted binary mask from the trained VISTA-XL checkpoint. No fake simulations.',
    },
    {
      icon: <Layers size={20} />, tag: 'Architecture',
      title: 'Hybrid Vision Model',
      body: 'Combines a pretrained Swin-Tiny encoder with an mLSTM bottleneck and a U-Net-style decoder for memory-efficient segmentation.',
    },
    {
      icon: <BarChart2 size={20} />, tag: 'Evaluation',
      title: 'Validated on ISIC 2017',
      body: 'Evaluated across 150 validation images. Mean Dice 0.8723 ± 0.12. Individual per-image metrics are not extrapolated from this result.',
    },
  ];

  return (
    <main className="page-content">
      {/* ─── Hero ─── */}
      <section className="hero-section" aria-labelledby="hero-heading">
        <div className="container">
          <div className="hero-section__grid">
            <div>
              <div className="hero-section__eyebrow" aria-hidden="true">
                Computer Vision / Research Lab
              </div>
              <h1 id="hero-heading" className="hero-headline">
                See beneath<br />the{' '}
                <mark>surface.</mark>
              </h1>
              <p className="hero-desc">
                VISTA-XL is a research prototype for skin-lesion segmentation using a Swin-Tiny encoder,
                mLSTM memory block, and U-Net-style decoder — trained on ISIC 2017 dermoscopy images.
                Upload an image to inspect the model's predicted segmentation mask.
              </p>
              <div className="hero-actions">
                <button
                  id="hero-analyze-btn"
                  className="btn btn-primary btn-lg"
                  onClick={() => navigate('/analyze')}
                >
                  Analyze an Image <ArrowRight size={18} />
                </button>
                <button
                  id="hero-model-btn"
                  className="btn btn-secondary btn-lg"
                  onClick={() => navigate('/model')}
                >
                  Explore the Model
                </button>
              </div>
              <p style={{ marginTop: 20, fontSize: '0.78rem', opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
                Research prototype only · Not a clinical diagnostic tool
              </p>
            </div>
            <HeroVisual />
          </div>
        </div>
      </section>

      {/* ─── Metrics strip ─── */}
      <section className="page-section" aria-label="Validation metrics summary">
        <div className="container">
          <div className="section-head">
            <div className="section-eyebrow">Validation Results — 150 Images (ISIC 2017)</div>
            <h2>Model Performance</h2>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 16 }}>
            {Object.values(VALIDATION_METRICS).map(m => (
              <MetricHighlight
                key={m.label}
                value={(m.value * 100).toFixed(1) + '%'}
                label={m.label}
                sub={m.description}
              />
            ))}
          </div>
          <p style={{ marginTop: 16, fontFamily: 'var(--font-mono)', fontSize: '0.68rem', opacity: 0.55 }}>
            ↑ Dataset-level metrics only. Dice and IoU per individual image require that image's ground-truth mask.
          </p>
        </div>
      </section>

      {/* ─── Feature cards ─── */}
      <section className="page-section" aria-label="Platform features">
        <div className="container">
          <div className="section-head">
            <div className="section-eyebrow">Platform Capabilities</div>
            <h2>What VISTA-XL does</h2>
          </div>
          <div className="research-grid">
            {featureCards.map(fc => (
              <div key={fc.title} className="research-card">
                <div className="research-card__icon">{fc.icon}</div>
                <div className="research-card__tag">{fc.tag}</div>
                <div className="research-card__title">{fc.title}</div>
                <p className="research-card__body">{fc.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── CTA teal band ─── */}
      <section
        className="page-section page-section--teal"
        aria-label="Call to action"
        style={{ borderBottom: 'var(--border)' }}
      >
        <div className="container text-center" style={{ maxWidth: 640, margin: '0 auto' }}>
          <div className="section-eyebrow section-eyebrow--light" style={{ justifyContent: 'center' }}>
            Try It Now
          </div>
          <h2 style={{ color: 'var(--ivory)', marginBottom: 16 }}>
            Upload a dermoscopic image
          </h2>
          <p style={{ color: 'rgba(242,242,235,0.8)', marginBottom: 32, fontSize: '1rem' }}>
            Get an actual predicted segmentation mask from {MODEL_INFO.name} — the Swin-Tiny + mLSTM model.
          </p>
          <button
            id="cta-analyze-btn"
            className="btn btn-primary btn-lg"
            onClick={() => navigate('/analyze')}
          >
            Start Analysis <ArrowRight size={18} />
          </button>
        </div>
      </section>
    </main>
  );
}
