import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// ─── API Configuration ────────────────────────────────────────────────────────
const API_URL = '/api';

export default function AuthPage({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setErrorMsg('');
    setUsername('');
    setPassword('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg('');

    try {
      if (isLogin) {
        // FastAPI OAuth2 uses form-data, not JSON for login by default
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        const response = await axios.post(`${API_URL}/login`, formData, {
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        
        const token = response.data.access_token;
        localStorage.setItem('jwtToken', token);
        onLogin(token);
      } else {
        // Signup expects JSON body
        const response = await axios.post(`${API_URL}/signup`, {
          username,
          password
        });
        
        const token = response.data.access_token;
        localStorage.setItem('jwtToken', token);
        onLogin(token);
      }
    } catch (err) {
      if (err.response && err.response.data) {
        setErrorMsg(err.response.data.detail || 'Authentication failed');
      } else {
        setErrorMsg('Network error. Ensure backend is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      {/* Background Decor */}
      <div style={styles.backgroundGrid} />

      <motion.div 
        style={styles.card}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      >
        {/* Left Side: Dynamic Content with Sliding Animation */}
        <div style={styles.contentArea}>
          <AnimatePresence mode="wait">
            <motion.div
              key={isLogin ? 'login' : 'signup'}
              initial={{ opacity: 0, x: isLogin ? -30 : 30 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: isLogin ? 30 : -30 }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              style={styles.formContainer}
            >
              <h2 style={styles.title}>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
              <p style={styles.subtitle}>
                {isLogin 
                  ? 'Access your personalized voice assistant.' 
                  : 'Join to experience native voice intelligence.'}
              </p>

              <form onSubmit={handleSubmit} style={styles.form}>
                <div style={styles.inputGroup}>
                  <label style={styles.label}>Username</label>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    style={styles.input}
                    required
                  />
                </div>
                <div style={styles.inputGroup}>
                  <label style={styles.label}>Password</label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    style={styles.input}
                    required
                  />
                </div>

                <AnimatePresence>
                  {errorMsg && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      style={styles.errorBox}
                    >
                      <span style={styles.errorText}>{errorMsg}</span>
                    </motion.div>
                  )}
                </AnimatePresence>

                <button 
                  type="submit" 
                  disabled={loading || !username || !password}
                  style={{
                    ...styles.submitBtn,
                    opacity: (loading || !username || !password) ? 0.7 : 1,
                    cursor: (loading || !username || !password) ? 'not-allowed' : 'pointer'
                  }}
                >
                  {loading 
                    ? 'Processing...' 
                    : (isLogin ? 'Sign In' : 'Sign Up')}
                </button>
              </form>

              <div style={styles.toggleContainer}>
                <span style={styles.toggleText}>
                  {isLogin ? "Don't have an account?" : "Already have an account?"}
                </span>
                <button 
                  type="button" 
                  onClick={toggleMode}
                  style={styles.toggleBtn}
                >
                  {isLogin ? 'Sign up' : 'Log in'}
                </button>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Right Side: Visual Accent Panel */}
        <motion.div 
          style={{
            ...styles.accentPanel,
            backgroundColor: isLogin ? '#000000' : '#f4f4f4',
            color: isLogin ? '#ffffff' : '#000000',
          }}
          animate={{
            backgroundColor: isLogin ? '#000000' : '#f4f4f4',
            color: isLogin ? '#ffffff' : '#000000',
          }}
          transition={{ duration: 0.6 }}
        >
          <motion.div
            key={isLogin ? 'login-accent' : 'signup-accent'}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            style={styles.accentContent}
          >
            <div style={styles.logo}>
              <svg width="40" height="40" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.39-.9.88 0 2.76-2.24 5-5.01 5s-5.01-2.24-5.01-5c0-.49-.41-.88-.9-.88s-.9.39-.9.88c0 3.41 2.72 6.23 6.06 6.72V21c0 .55.45 1 1 1s1-.45 1-1v-2.4c3.34-.49 6.06-3.31 6.06-6.72 0-.49-.41-.88-.9-.88z"/>
              </svg>
            </div>
            <h3 style={styles.accentTitle}>
              {isLogin ? 'Nova AI' : 'Intelligence'}
            </h3>
            <p style={styles.accentText}>
              {isLogin 
                ? 'Your secure connection to next-generation voice processing.' 
                : 'Experience zero-latency bidirectional streaming architecture.'}
            </p>
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}

// ─── STYLES ──────────────────────────────────────────────────────────────────
const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
    position: 'relative',
    overflow: 'hidden',
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  backgroundGrid: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    backgroundImage: `radial-gradient(#e5e5e5 1px, transparent 1px)`,
    backgroundSize: '32px 32px',
    opacity: 0.4,
    zIndex: 0,
  },
  card: {
    width: '900px',
    height: '600px',
    backgroundColor: '#ffffff',
    borderRadius: '24px',
    display: 'flex',
    flexDirection: 'row',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(0,0,0,0.05)',
    zIndex: 1,
    overflow: 'hidden',
  },
  contentArea: {
    flex: 1,
    padding: '60px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    backgroundColor: '#fafafa',
  },
  formContainer: {
    width: '100%',
    maxWidth: '380px',
    margin: '0 auto',
  },
  title: {
    fontSize: '36px',
    fontWeight: 800,
    color: '#000000',
    margin: '0 0 8px 0',
    letterSpacing: '-1px',
  },
  subtitle: {
    fontSize: '15px',
    color: '#666666',
    margin: '0 0 40px 0',
    lineHeight: '1.5',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  label: {
    fontSize: '13px',
    fontWeight: 600,
    color: '#333333',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  input: {
    padding: '16px',
    fontSize: '16px',
    borderRadius: '12px',
    border: '1px solid #e0e0e0',
    backgroundColor: '#ffffff',
    color: '#000000',
    outline: 'none',
    transition: 'all 0.2s',
    boxShadow: '0 2px 4px rgba(0,0,0,0.02) inset',
  },
  errorBox: {
    padding: '12px 16px',
    backgroundColor: '#fff0f0',
    borderLeft: '4px solid #ff4444',
    borderRadius: '4px',
    marginTop: '-8px',
  },
  errorText: {
    fontSize: '14px',
    color: '#cc0000',
    fontWeight: 500,
  },
  submitBtn: {
    marginTop: '8px',
    padding: '16px',
    fontSize: '16px',
    fontWeight: 600,
    color: '#ffffff',
    backgroundColor: '#000000',
    border: 'none',
    borderRadius: '12px',
    transition: 'transform 0.1s, background-color 0.2s',
  },
  toggleContainer: {
    marginTop: '32px',
    textAlign: 'center',
    fontSize: '14px',
  },
  toggleText: {
    color: '#666666',
  },
  toggleBtn: {
    background: 'none',
    border: 'none',
    color: '#000000',
    fontWeight: 700,
    marginLeft: '6px',
    cursor: 'pointer',
    padding: 0,
    textDecoration: 'underline',
  },
  accentPanel: {
    flex: '0.8',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    padding: '60px',
    position: 'relative',
  },
  accentContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'baseline',
  },
  logo: {
    marginBottom: '24px',
  },
  accentTitle: {
    fontSize: '32px',
    fontWeight: 700,
    margin: '0 0 16px 0',
    lineHeight: '1.2',
  },
  accentText: {
    fontSize: '16px',
    lineHeight: '1.6',
    opacity: 0.8,
  }
};
