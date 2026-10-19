import { Microscope } from 'lucide-react';
import { NavLink } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="footer" role="contentinfo">
      <div className="container">
        <div className="footer__grid">
          <div>
            <div className="footer__brand">VISTA-XL</div>
            <p className="footer__desc">
              Vision xLSTM-Enhanced Swin-UNet for binary skin-lesion segmentation.
              A computer-vision research project — not a clinically validated diagnostic tool.
            </p>
            <div className="flex gap-8 mt-16">
              <span className="label label-yellow">
                <Microscope size={10} /> Research Prototype
              </span>
            </div>
          </div>
          <div>
            <div className="footer__heading">Navigation</div>
            <ul className="footer__links">
              <li><NavLink to="/">Home</NavLink></li>
              <li><NavLink to="/analyze">Analyze Image</NavLink></li>
              <li><NavLink to="/model">Model Information</NavLink></li>
              <li><NavLink to="/research">Research Overview</NavLink></li>
            </ul>
          </div>
          <div>
            <div className="footer__heading">Resources</div>
            <ul className="footer__links">
              <li><a href="https://github.com/NX-AI/xlstm" target="_blank" rel="noreferrer">xLSTM Paper</a></li>
              <li><a href="https://www.isic-archive.com/" target="_blank" rel="noreferrer">ISIC Archive</a></li>
              <li><a href="https://github.com/microsoft/Swin-Transformer" target="_blank" rel="noreferrer">Swin Transformer</a></li>
            </ul>
          </div>
        </div>
        <div className="footer__bottom">
          <span className="footer__copyright">
            © 2026 VISTA-XL Research Project. Predictions are not medical diagnoses.
          </span>
          <a
            href="https://github.com"
            aria-label="GitHub repository"
            className="flex items-center gap-8 opacity-60"
            style={{ color: 'var(--ivory)', fontSize: '0.85rem' }}
          >

          </a>
        </div>
      </div>
    </footer>
  );
}
