import {
  Microscope, Database, BarChart2, AlertOctagon,
  FlaskConical, Layers, ArrowRight, BookOpen
} from 'lucide-react';

const cards = [
  {
    icon: <Microscope size={20} />,
    tag: 'Problem Statement',
    title: 'Skin Lesion Segmentation',
    body: `Dermoscopy-based skin-lesion segmentation is a binary pixel-classification task: assign each
pixel to "lesion" or "background". Accurate delineation supports clinical measurement of lesion area,
asymmetry, and boundary irregularity — but predictions from a research model are not a clinical diagnosis.`,
  },
  {
    icon: <Database size={20} />,
    tag: 'Dataset',
    title: 'ISIC 2017 (Likely)',
    body: `The dataset follows the ISIC_XXXXXXX naming convention and contains 2,000 training and 150
validation image–mask pairs. This matches the official ISIC 2017 Skin Lesion Segmentation Challenge
split, though dataset provenance metadata is not locally present. Masks are clean binary (values 0/255).
Resolutions vary: training ≈ 1022×767 px, validation ≈ 3024×2016 px.`,
  },
  {
    icon: <Layers size={20} />,
    tag: 'Architecture',
    title: 'Swin-Tiny + mLSTM + U-Net Decoder',
    body: `VISTA-XL encodes input images at 224×224 px through a pretrained Swin-Tiny producing four
hierarchical feature maps (f1–f4). The deepest map f4 (B×768×7×7) is flattened into 49 tokens and
processed by an mLSTM block (embedding=768, heads=4, context=49) for long-range dependency
modelling. The decoder progressively upsamples with skip connections from f1–f3, then a 1×1 conv
produces a single-channel logit map.`,
  },
  {
    icon: <FlaskConical size={20} />,
    tag: 'Training',
    title: 'Preprocessing & Training',
    body: `Images are resized bilinearly to 224×224 and normalized with ImageNet statistics
(μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]). Loss = Binary Cross-Entropy + Soft Dice.
Optimizer: AdamW (lr=1e-4, wd=1e-5). Trained for 10 epochs, batch size 8.
Best checkpoint selected on validation Dice.`,
  },
  {
    icon: <BarChart2 size={20} />,
    tag: 'Evaluation Metrics',
    title: 'Dice & IoU',
    body: `Segmentation quality is measured with Dice coefficient (2·TP / (2·TP + FP + FN)) and Intersection
over Union (TP / (TP + FP + FN)), both computed per image then averaged. Precision, recall,
specificity, and pixel accuracy are also reported. Ground-truth masks are required; predictions alone
do not yield Dice or IoU values.`,
  },
  {
    icon: <AlertOctagon size={20} />,
    tag: 'Limitations',
    title: 'Research Boundaries',
    body: `No independent test set is available locally; results reflect validation performance only.
Training used no mixed-precision on CPU. The mLSTM block processes spatial tokens sequentially
and may scale poorly to high-resolution inputs. Dataset provenance is unconfirmed and no clinical
validation has been performed. This tool must not be used for diagnostic decisions.`,
  },
];

const timeline = [
  { step: '01', label: 'Upload', desc: 'Researcher uploads a dermoscopic PNG/JPG/BMP (≤20 MB).' },
  { step: '02', label: 'Preprocess', desc: 'Backend resizes to 224×224, applies ImageNet normalization, converts to torch tensor.' },
  { step: '03', label: 'Encode', desc: 'Swin-Tiny produces four feature maps f1–f4 at progressively halved spatial resolution.' },
  { step: '04', label: 'mLSTM', desc: 'f4 tokens passed through mLSTM block for sequential memory processing.' },
  { step: '05', label: 'Decode', desc: 'U-Net decoder upsamples with skip connections → single logit map → sigmoid probability.' },
  { step: '06', label: 'Threshold', desc: 'Pixels with probability ≥ 0.5 classified as lesion → binary mask returned.' },
  { step: '07', label: 'Visualize', desc: 'Original, binary mask, and coral overlay (α=0.42) displayed to researcher.' },
];

export default function ResearchPage() {
  return (
    <main className="page-content">
      <div className="container" style={{ paddingTop: 48, paddingBottom: 80 }}>

        {/* Header */}
        <div style={{ marginBottom: 56 }}>
          <div className="section-eyebrow">Research Overview</div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3.2rem)', marginBottom: 12 }}>
            Methodology & Context
          </h1>
          <p style={{ maxWidth: 640, opacity: 0.75, fontSize: '1.05rem' }}>
            An editorial overview of the problem, dataset, architecture, training methodology,
            evaluation metrics, and known limitations of the VISTA-XL research project.
          </p>
        </div>

        {/* Research cards */}
        <div className="research-grid" style={{ marginBottom: 64 }}>
          {cards.map(c => (
            <div key={c.title} className="research-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div className="research-card__icon">{c.icon}</div>
                <span className="research-card__tag">{c.tag}</span>
              </div>
              <div className="research-card__title">{c.title}</div>
              <p className="research-card__body" style={{ whiteSpace: 'pre-line' }}>{c.body}</p>
            </div>
          ))}
        </div>

        {/* Segmentation workflow */}
        <div style={{ marginBottom: 64 }}>
          <div className="section-eyebrow" style={{ marginBottom: 8 }}>
            <BookOpen size={12} /> Segmentation Workflow
          </div>
          <h2 style={{ marginBottom: 32 }}>End-to-End Inference Pipeline</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {timeline.map((t, i) => (
              <div
                key={t.step}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '60px 1fr',
                  gap: 0,
                  position: 'relative',
                }}
              >
                {/* Step number column */}
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                }}>
                  <div style={{
                    width: 40,
                    height: 40,
                    background: i % 2 === 0 ? 'var(--teal)' : 'var(--coral)',
                    border: 'var(--border)',
                    borderRadius: '3px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: 'var(--ivory)',
                    flexShrink: 0,
                    zIndex: 1,
                  }}>
                    {t.step}
                  </div>
                  {i < timeline.length - 1 && (
                    <div style={{ width: 2, flex: 1, background: 'var(--ink)', minHeight: 24, opacity: 0.2 }} />
                  )}
                </div>
                {/* Content */}
                <div style={{ padding: '4px 0 28px 20px' }}>
                  <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.95rem', marginBottom: 4 }}>
                    {t.label}
                  </div>
                  <p style={{ fontSize: '0.85rem', opacity: 0.75 }}>{t.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Future research */}
        <div style={{
          background: 'var(--ink)',
          color: 'var(--ivory)',
          border: 'var(--border)',
          borderRadius: 'var(--radius)',
          padding: '36px 40px',
          boxShadow: 'var(--shadow-hard)',
        }}>
          <div className="section-eyebrow section-eyebrow--light" style={{ marginBottom: 12 }}>
            Future Directions
          </div>
          <h2 style={{ color: 'var(--ivory)', marginBottom: 20 }}>Open Research Questions</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 16 }}>
            {[
              'Acquire independent ISIC 2017 test set (600 images) for unbiased evaluation',
              'Controlled ablation: VISTA-XL vs. Swin baseline vs. U-Net baseline on identical splits',
              'Explore mLSTM at shallower decoder levels (multi-scale memory)',
              'Test on ISIC 2018 / PH2 for out-of-distribution generalization',
              'Mixed-precision training & GPU acceleration for scale experiments',
              'Confirm dataset provenance and document image-mask alignment audit',
            ].map(q => (
              <div key={q} style={{
                background: 'rgba(242,242,235,0.06)',
                border: '1px solid rgba(242,242,235,0.12)',
                borderRadius: '3px',
                padding: '14px 16px',
                fontSize: '0.85rem',
                lineHeight: 1.6,
                color: 'rgba(242,242,235,0.82)',
                display: 'flex',
                gap: 10,
                alignItems: 'flex-start',
              }}>
                <ArrowRight size={14} style={{ flexShrink: 0, marginTop: 3, color: 'var(--coral)' }} />
                {q}
              </div>
            ))}
          </div>
        </div>

      </div>
    </main>
  );
}
