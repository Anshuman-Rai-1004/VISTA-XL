import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AppNavbar from './components/AppNavbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import AnalyzePage from './pages/AnalyzePage';
import ModelPage from './pages/ModelPage';
import ResearchPage from './pages/ResearchPage';

export default function App() {
  return (
    <BrowserRouter>
      <AppNavbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/analyze" element={<AnalyzePage />} />
        <Route path="/model" element={<ModelPage />} />
        <Route path="/research" element={<ResearchPage />} />
        {/* Fallback */}
        <Route path="*" element={<HomePage />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}
