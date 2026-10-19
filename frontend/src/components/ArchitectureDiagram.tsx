export default function ArchitectureDiagram() {
  const blocks = [
    {
      label: 'Input Image',
      sub: 'B × 3 × 224 × 224',
      cls: '',
      bg: 'var(--offwhite)',
    },
    null, // connector
    {
      label: 'Swin-Tiny Encoder',
      sub: 'Hierarchical ViT · 4 stages\nf1: 96ch · f2: 192ch · f3: 384ch · f4: 768ch',
      cls: 'arch-box--encoder',
      bg: '',
    },
    null,
    {
      label: 'mLSTM Bottleneck',
      sub: 'matrix-LSTM block\nembedding=768 · heads=4 · context=49\napplied to f4 tokens (7×7)',
      cls: 'arch-box--mlstm',
      bg: '',
    },
    null,
    {
      label: 'U-Net Decoder',
      sub: 'decode3: [768+384→256] · decode2: [256+192→128]\ndecode1: [128+96→64] · decode0: [64→32] · refine: [32→16]',
      cls: 'arch-box--decoder',
      bg: '',
    },
    null,
    {
      label: 'Output Head',
      sub: 'Conv1×1 → B × 1 × 224 × 224\nSigmoid → threshold 0.5 → Binary mask',
      cls: 'arch-box--output',
      bg: '',
    },
  ];

  const connectorLabels = [
    'RGB tensor',
    'Skip connections (f1, f2, f3) + deep features (f4)',
    'Processed f4 tokens',
    'Progressive upsampling + skip concat',
    'Logits → sigmoid',
  ];

  let connIdx = 0;

  return (
    <div className="arch-diagram" role="img" aria-label="VISTA-XL architecture flow diagram">
      {blocks.map((block, i) => {
        if (block === null) {
          const label = connectorLabels[connIdx++] ?? '';
          return (
            <div key={`conn-${i}`} className="arch-connector">
              <span className="arch-connector__label">{label}</span>
              <div className="arch-arrow" style={{ position: 'absolute', bottom: -1 }} />
            </div>
          );
        }
        return (
          <div key={`block-${i}`} className="arch-block">
            <div
              className={`arch-box ${block.cls}`}
              style={block.bg ? { background: block.bg } : {}}
            >
              <div className="arch-box__label">{block.label}</div>
              {block.sub.split('\n').map((line, j) => (
                <div key={j} className="arch-box__sub">{line}</div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
