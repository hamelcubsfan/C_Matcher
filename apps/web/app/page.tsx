'use client';

import Image from 'next/image';
import { ChangeEvent, FormEvent, useMemo, useState } from 'react';

import { API_BASE } from '../lib/apiBase';

type CandidateRead = {
  id: string;
  full_name: string | null;
  email: string | null;
  location: string | null;
  resume_url: string | null;
  parsed_profile: Record<string, unknown>;
  skills: string[];
  created_at: string;
  updated_at: string;
};

type JobRead = {
  id: number;
  greenhouse_job_id: string;
  title: string;
  team: string | null;
  location: string | null;
  must_have_skills: string[];
  nice_to_have_skills: string[];
  description: string | null;
  absolute_url: string | null;
  posting_status: string;
};

type MatchResult = {
  job: JobRead;
  retrieval_score?: number | null;
  rerank_score?: number | null;
  confidence?: number | null;
  explanation?: string | null;
  reason_codes?: Record<string, unknown>[];
};

type BannerState = {
  type: 'success' | 'error';
  message: string;
} | null;

const formatScore = (value?: number | null) => {
  if (value === null || value === undefined) {
    return '—';
  }
  return value.toFixed(2);
};

const confidenceBadge = (value?: number | null) => {
  if (value === null || value === undefined) {
    return { label: 'Confidence pending', className: 'badge confidence-medium' };
  }
  if (value >= 0.75) {
    return { label: 'High confidence', className: 'badge confidence-high' };
  }
  if (value >= 0.4) {
    return { label: 'Medium confidence', className: 'badge confidence-medium' };
  }
  return { label: 'Emerging fit', className: 'badge confidence-low' };
};

const normaliseReason = (reason: Record<string, unknown>) => {
  if (!reason) return '';
  const maybeLabel = typeof reason.label === 'string' ? reason.label : undefined;
  const maybeCode = typeof reason.code === 'string' ? reason.code : undefined;
  const label = maybeLabel ?? maybeCode?.replace(/_/g, ' ').toLowerCase();
  const score = typeof reason.score === 'number' ? Math.round(reason.score * 100) : undefined;
  const weight = typeof reason.weight === 'number' ? Math.round(reason.weight * 100) : undefined;
  const suffix = score !== undefined ? `${score}%` : weight !== undefined ? `${weight}%` : undefined;
  if (label && suffix) {
    return `${label} · ${suffix}`;
  }
  if (label) {
    return label;
  }
  const entries = Object.entries(reason)
    .filter(([, value]) => typeof value === 'string' || typeof value === 'number')
    .slice(0, 2)
    .map(([key, value]) => `${key}: ${value}`);
  return entries.join(' · ');
};

const toDisplayDate = (value: string | null) => {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString();
};

import ReactMarkdown from 'react-markdown';

// ... (keep types the same)

const LOADING_MESSAGES = [
  "Expanding search queries...",
  "Scanning 50+ job descriptions...",
  "Analyzing semantic fit with Gemini...",
  "Checking for seniority mismatch...",
  "Verifying technical domain expertise...",
  "Ranking top candidates...",
  "Finalizing hiring pitch...",
];

function LoadingMessages() {
  const [index, setIndex] = useState(0);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return <span>{LOADING_MESSAGES[index]}</span>;
}

// Sensor Ring Component
function SensorRing({ value, label, size = 60 }: { value: number | null | undefined; label: string; size?: number }) {
  const score = value ? Math.round(value * 100) : 0;
  const radius = (size - 8) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (score / 100) * circumference;

  let color = '#E5E7EB';
  if (score >= 75) color = '#00E89D'; // Green
  else if (score >= 40) color = '#F59E0B'; // Yellow
  else if (score > 0) color = '#EF4444'; // Red

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
      <div style={{ position: 'relative', width: size, height: size }}>
        {/* Background Circle */}
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#E5E7EB"
            strokeWidth="4"
          />
          {/* Progress Circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="4"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 1s ease-out' }}
          />
        </svg>
        {/* Score Text */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          fontSize: '0.9rem',
          fontWeight: 700,
          color: 'var(--waymo-navy)'
        }}>
          {value !== null && value !== undefined ? score : '—'}
        </div>
      </div>
      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </span>
    </div>
  );
}

// Skeleton Card Component
function SkeletonCard() {
  return (
    <div className="match-card" style={{ height: '200px', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ width: '60%', height: '24px', background: '#F3F4F6', borderRadius: '4px', animation: 'pulse 1.5s infinite' }}></div>
      <div style={{ width: '40%', height: '16px', background: '#F3F4F6', borderRadius: '4px', animation: 'pulse 1.5s infinite' }}></div>
      <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto' }}>
        <div style={{ width: '60px', height: '60px', borderRadius: '50%', background: '#F3F4F6', animation: 'pulse 1.5s infinite' }}></div>
        <div style={{ width: '60px', height: '60px', borderRadius: '50%', background: '#F3F4F6', animation: 'pulse 1.5s infinite' }}></div>
        <div style={{ width: '60px', height: '60px', borderRadius: '50%', background: '#F3F4F6', animation: 'pulse 1.5s infinite' }}></div>
      </div>
    </div>
  );
}

import React from 'react';

export default function HomePage() {
  const [inputMode, setInputMode] = useState<'upload' | 'paste'>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [pasteText, setPasteText] = useState('');
  const [notes, setNotes] = useState('');
  const [candidate, setCandidate] = useState<CandidateRead | null>(null);
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loadingMatches, setLoadingMatches] = useState(false);
  const [banner, setBanner] = useState<BannerState>(null);

  const [fileError, setFileError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setFileError(null);
      setBanner(null);
    }
  };

  const handleUpload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (inputMode === 'upload' && !selectedFile) {
      setFileError('Choose a resume file to upload.');
      return;
    }
    if (inputMode === 'paste' && !pasteText.trim()) {
      setFileError('Paste resume text to proceed.');
      return;
    }

    setUploading(true);
    setBanner(null);

    try {
      let response;
      if (inputMode === 'upload' && selectedFile) {
        const form = new FormData();
        form.append('file', selectedFile);
        if (notes.trim()) {
          form.append('notes', notes.trim());
        }
        response = await fetch(`${API_BASE}/candidates/upload`, {
          method: 'POST',
          body: form,
        });
      } else {
        response = await fetch(`${API_BASE}/candidates/paste`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: pasteText, notes: notes.trim() || null }),
        });
      }

      if (!response.ok) {
        let detail = 'Upload failed. Check the API logs for more detail.';
        try {
          const payload = await response.json();
          detail = typeof payload.detail === 'string' ? payload.detail : detail;
        } catch (error) {
          /* ignore JSON parsing issues */
        }
        throw new Error(detail);
      }
      const data = (await response.json()) as { candidate: CandidateRead };
      setCandidate(data.candidate);
      setMatches([]);
      setBanner({ type: 'success', message: 'Resume processed. Request matches to see the strongest Waymo roles.' });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Something went wrong during upload.';
      setBanner({ type: 'error', message });
    } finally {
      setUploading(false);
    }
  };

  const [progress, setProgress] = useState(0);
  const [loadingMessage, setLoadingMessage] = useState('');

  const requestMatches = async () => {
    if (!candidate) {
      setBanner({ type: 'error', message: 'Upload a resume before requesting matches.' });
      return;
    }
    setLoadingMatches(true);
    setBanner(null);
    setProgress(0);
    setLoadingMessage('Initializing search...');
    setMatches([]);

    try {
      const response = await fetch(`${API_BASE}/match/${candidate.id}?limit=5`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Match request failed: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('No response body received');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.status) setLoadingMessage(data.status);
              if (typeof data.progress === 'number') setProgress(data.progress);

              if (data.data) {
                // Final result
                setMatches(data.data);
                if (!data.data.length) {
                  setBanner({ type: 'success', message: 'No immediate matches yet. Try updating the resume or ingesting more jobs.' });
                }
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error fetching matches.';
      setBanner({ type: 'error', message });
    } finally {
      setLoadingMatches(false);
      setProgress(0);
    }
  };

  const displayedSkills = useMemo(() => {
    if (!candidate?.skills?.length) return [] as string[];
    return candidate.skills.slice(0, 10);
  }, [candidate]);

  return (
    <main>
      <section className="hero-section">
        <div className="logo-row">
          <Image
            src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Waymo_logo.svg"
            alt="Waymo logo"
            width={60}
            height={60}
            priority
          />
        </div>
        <div className="logo-title">Waymo Role Matcher</div>
        <p className="subtitle">
          Match candidates to teams in seconds.
        </p>
      </section>

      <section className="card" style={{ marginTop: '2rem' }}>
        <h2>1. Add a candidate resume</h2>
        <div className="tabs">
          <button
            className={`tab-button ${inputMode === 'upload' ? 'active' : ''}`}
            onClick={() => { setInputMode('upload'); setFileError(null); }}
          >
            Upload File
          </button>
          <button
            className={`tab-button ${inputMode === 'paste' ? 'active' : ''}`}
            onClick={() => { setInputMode('paste'); setFileError(null); }}
          >
            Paste Text
          </button>
        </div>

        <form onSubmit={handleUpload}>
          {inputMode === 'upload' ? (
            <>
              {/* Upload Section */}
              <div className="card" style={{ textAlign: 'center' }}>
                <div className="upload-section">
                  <label className="upload-label">
                    <input
                      type="file"
                      accept=".txt,.pdf,.md"
                      onChange={handleFileChange}
                      disabled={uploading}
                    />
                    <div className="upload-icon-wrapper">
                      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                    </div>
                    <span className="upload-text-main">Click to upload resume</span>
                    <span className="upload-text-sub">PDF, TXT, or Markdown</span>
                  </label>
                </div>

                <div style={{ marginTop: '2rem', textAlign: 'left' }}>
                  <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>Recruiter Notes (Optional)</h3>
                  <textarea
                    className="notes-input"
                    placeholder="Add context: 'Strong on ML, prefers Seattle', 'Met at conference', etc."
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    rows={3}
                    style={{ width: '100%', resize: 'vertical' }}
                  />
                </div>

                <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'center' }}>
                  <button
                    className="primary-button"
                    type="submit"
                    disabled={!selectedFile || uploading}
                  >
                    {uploading ? (
                      <span className="loading-text">
                        <div className="spinner"></div>
                        Scanning...
                      </span>
                    ) : (
                      'Analyze Candidate'
                    )}
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <textarea
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
                placeholder="Paste resume text here..."
                style={{ width: '100%', minHeight: '200px', padding: '1rem', borderRadius: '8px', border: '1px solid #ccc', fontFamily: 'monospace' }}
              />
              <div style={{ marginTop: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
                  Recruiter Notes (Optional)
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add context: 'Strong on ML, prefers Seattle', 'Met at conference', etc."
                  style={{ width: '100%', minHeight: '80px', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ccc', fontFamily: 'inherit' }}
                />
              </div>
              <div className="actions" style={{ marginTop: '1.5rem' }}>
                <button className="primary-button" type="submit" disabled={uploading}>
                  {uploading ? 'Processing…' : 'Process text'}
                </button>
                <small style={{ marginLeft: '1rem', color: 'var(--text-muted)' }}>We&apos;ll index the resume with Gemini embeddings for retrieval.</small>
              </div>
            </>
          )}

          {fileError && <div className="alert error" style={{ marginTop: '1rem' }}>{fileError}</div>}
        </form>
      </section>

      {candidate && (
        <section className="card" style={{ marginTop: '1.75rem' }}>
          <h2>2. Review candidate context</h2>
          <div className="candidate-overview">
            <div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.3rem' }}>
                Candidate Profile
              </div>
              <div style={{ color: 'var(--text-muted)' }}>
                Contact Hidden
                {candidate.location ? ` • ${candidate.location}` : ''}
              </div>
              <div style={{ color: 'var(--text-muted)', marginTop: '0.35rem', fontSize: '0.85rem' }}>
                Uploaded {toDisplayDate(candidate.created_at)}
              </div>
            </div>
            <div style={{ minWidth: '200px', textAlign: 'right' }}>
              {loadingMatches ? (
                <div style={{ textAlign: 'left' }}>
                  <div style={{ marginBottom: '0.5rem', fontSize: '0.9rem', fontWeight: 500, color: 'var(--waymo-teal)' }}>
                    {loadingMessage}
                  </div>
                  <div className="progress-container">
                    {/* Technical Waypoints (Ticks) */}
                    {[0, 25, 50, 75, 100].map((point) => (
                      <div
                        key={point}
                        style={{
                          position: 'absolute',
                          left: `${point}%`,
                          top: '50%',
                          transform: 'translate(-50%, -50%)',
                          width: '8px',
                          height: '8px',
                          background: progress >= point ? 'var(--waymo-blue)' : '#E5E7EB',
                          borderRadius: '50%',
                          zIndex: 1,
                          transition: 'all 0.3s',
                          boxShadow: progress >= point ? '0 0 0 2px white' : 'none'
                        }}
                      />
                    ))}

                    <div className="progress-bar" style={{ width: `${progress}%` }}></div>

                    <div className="waymo-car" style={{ left: `${progress}%` }}>
                      <div className="lidar-pulse"></div>
                      {/* Sharp Side Profile I-PACE */}
                      <svg width="40" height="20" viewBox="0 0 40 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 12C2 12 5 6 14 6H26C32 6 36 8 38 12V15C38 16.1 37.1 17 36 17H4C2.9 17 2 16.1 2 15V12Z" fill="white" stroke="#1D1D1F" strokeWidth="1.5" />
                        <circle cx="10" cy="17" r="3" fill="#1D1D1F" />
                        <circle cx="30" cy="17" r="3" fill="#1D1D1F" />
                        <rect x="18" y="3" width="4" height="3" fill="#1D1D1F" /> {/* Lidar Puck */}
                        <circle cx="20" cy="3" r="1" fill="#00E89D" /> {/* Active Green Dot */}
                      </svg>
                    </div>
                  </div>
                </div>
              ) : (
                <button
                  className="primary-button"
                  type="button"
                  onClick={requestMatches}
                >
                  Get top matches
                </button>
              )}
            </div>
          </div>
          <div style={{ marginTop: '1.5rem' }}>
            <h3 style={{ marginBottom: '0.75rem' }}>Detected skills</h3>
            {displayedSkills.length ? (
              <div className="skill-list">
                {displayedSkills.map((skill) => (
                  <span key={skill} className="skill-pill">
                    {skill}
                  </span>
                ))}
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)' }}>No skills extracted yet. Try a richer resume.</div>
            )}
          </div>
        </section>
      )}

      {banner && (
        <div className={`alert ${banner.type === 'error' ? 'error' : 'success'}`} style={{ marginTop: '1.5rem' }}>
          {banner.message}
        </div>
      )}

      {(matches.length > 0 || loadingMatches) && (
        <section className="card" style={{ marginTop: '1.75rem' }}>
          <h2>3. Route to the right Waymo roles</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
            Each card blends retrieval scores, reranker feedback, and Gemini explanations so you can take action fast.
          </p>

          <div className="match-grid">
            {loadingMatches ? (
              <>
                <SkeletonCard />
                <SkeletonCard />
                <SkeletonCard />
              </>
            ) : (
              matches.map((match) => {
                const badge = confidenceBadge(match.confidence);
                return (
                  <article key={match.job.id} className="match-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                      <div>
                        <h3 style={{ marginBottom: '0.35rem' }}>{match.job.title}</h3>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                          <span style={{ fontWeight: 600, marginRight: '0.5rem' }}>#{match.job.greenhouse_job_id}</span>
                          {match.job.team ?? 'Team TBD'} {match.job.location ? `• ${match.job.location}` : ''}
                        </div>
                      </div>
                      <span className={badge.className}>{badge.label}</span>
                    </div>

                    {/* Sensor Rings Row */}
                    <div style={{ display: 'flex', gap: '2rem', margin: '1.5rem 0', padding: '1rem 0', borderTop: '1px solid #F3F4F6', borderBottom: '1px solid #F3F4F6' }}>
                      <SensorRing value={match.retrieval_score} label="Retrieval" />
                      <SensorRing value={match.rerank_score} label="Rerank" />
                      <SensorRing value={match.confidence} label="Confidence" />
                    </div>

                    {match.explanation && (
                      <div className="markdown-content" style={{ color: 'var(--text-muted)', lineHeight: 1.55 }}>
                        <ReactMarkdown>{match.explanation}</ReactMarkdown>
                      </div>
                    )}
                    {match.reason_codes && match.reason_codes.length > 0 && (
                      <div style={{ marginTop: '1rem' }}>
                        <div style={{ fontSize: '0.8rem', fontWeight: 700, marginBottom: '0.5rem', color: 'var(--text-muted)' }}>
                          Why it surfaced
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                          {match.reason_codes
                            .map((reason, index) => ({ reason, label: normaliseReason(reason), index }))
                            .filter((entry) => entry.label)
                            .map((entry) => (
                              <span key={entry.index} className="reason-chip">
                                {entry.label}
                              </span>
                            ))}
                        </div>
                      </div>
                    )}
                    {match.job.absolute_url && (
                      <a
                        href={match.job.absolute_url}
                        target="_blank"
                        rel="noreferrer"
                        className="arrow-link"
                        style={{ marginTop: '1.5rem' }}
                      >
                        Open job posting
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M3.33337 8H12.6667" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M8 3.33331L12.6667 7.99998L8 12.6666" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </a>
                    )}
                  </article>
                );
              })
            )}
          </div>
        </section>
      )}
    </main>
  );
}
