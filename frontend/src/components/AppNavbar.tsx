import { useState, useEffect, useRef } from 'react';
import { Menu, X } from 'lucide-react';
import { NavLink, useNavigate } from 'react-router-dom';

export default function AppNavbar() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    function handle(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, []);

  const links = [
    { to: '/', label: 'Home' },
    { to: '/analyze', label: 'Analyze Image' },
    { to: '/model', label: 'Model Info' },
    { to: '/research', label: 'Research' },
  ];

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="container navbar__inner" ref={menuRef}>
        {/* Wordmark */}
        <NavLink to="/" className="navbar__wordmark" aria-label="VISTA-XL home">
          <span className="navbar__wordmark-dot" aria-hidden="true" />
          VISTA-XL
        </NavLink>

        {/* Desktop nav links */}
        <ul className={`navbar__links${open ? ' open' : ''}`} role="list">
          {links.map(({ to, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) => isActive ? 'active' : ''}
                onClick={() => setOpen(false)}
              >
                {label}
              </NavLink>
            </li>
          ))}
          <li>
            <button
              className="navbar__cta"
              onClick={() => { navigate('/analyze'); setOpen(false); }}
            >
              New Analysis
            </button>
          </li>
        </ul>

        {/* Mobile hamburger */}
        <button
          className="navbar__hamburger"
          aria-label={open ? 'Close menu' : 'Open menu'}
          aria-expanded={open}
          onClick={() => setOpen(v => !v)}
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>
    </nav>
  );
}
