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
              <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                The API stores a text-only copy for matching. PDFs work best, but plain text is supported too.
              </p>
              <label className="upload-label">
                <span role="img" aria-hidden="true">
                  📄
                </span>
                <span>{selectedFile ? selectedFile.name : 'Choose a resume (PDF or text)'}</span>
                <input type="file" accept=".pdf,.txt,.md,.doc,.docx,.rtf" onChange={handleFileChange} />
              </label>
            </>
          ) : (
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              placeholder="Paste resume text here..."
              style={{ width: '100%', minHeight: '200px', padding: '1rem', borderRadius: '8px', border: '1px solid #ccc', fontFamily: 'monospace' }}
            />
          )}

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

          {fileError && <div className="alert error" style={{ marginTop: '1rem' }}>{fileError}</div>}
          <div className="actions" style={{ marginTop: '1.5rem' }}>
            <button className="primary-button" type="submit" disabled={uploading}>
              {uploading ? 'Processing…' : (inputMode === 'upload' ? 'Upload resume' : 'Process text')}
            </button>
            <small style={{ marginLeft: '1rem', color: 'var(--text-muted)' }}>We&apos;ll index the resume with Gemini embeddings for retrieval.</small>
          </div>
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
                    <div className="progress-bar" style={{ width: `${progress}%` }}></div>
                    <div className="waymo-car" style={{ left: `${progress}%` }}>
                      <svg width="60" height="30" viewBox="0 0 60 30" fill="none" xmlns="http://www.w3.org/2000/svg">
                        {/* Jaguar I-PACE Silhouette */}
                        <path d="M2 20C2 20 5 12 18 12H38C46 12 52 15 56 20V24C56 26.2 54.2 28 52 28H8C5.8 28 4 26.2 4 24L2 20Z" fill="white" stroke="#1d1d1f" strokeWidth="2" />

                        {/* Waymo Sensor Dome */}
                        <path d="M26 6H34V12H26V6Z" fill="#1d1d1f" />
                        <circle cx="30" cy="5" r="3" fill="#1d1d1f" />
                        <circle cx="30" cy="5" r="1.5" fill="#3ddc91" /> {/* Lidar Spin */}

                        {/* Wheels */}
                        <circle cx="14" cy="28" r="5" fill="#1d1d1f" />
                        <circle cx="46" cy="28" r="5" fill="#1d1d1f" />
                        <circle cx="14" cy="28" r="2" fill="#555" />
                        <circle cx="46" cy="28" r="2" fill="#555" />

                        {/* Windows */}
                        <path d="M16 14L20 14L22 20H14L16 14Z" fill="#0056f5" fillOpacity="0.2" />
                        <path d="M24 14H36L38 20H22L24 14Z" fill="#0056f5" fillOpacity="0.2" />
                        <path d="M38 14H44L42 20H38V14Z" fill="#0056f5" fillOpacity="0.2" />
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

      {matches.length > 0 && (
        <section className="card" style={{ marginTop: '1.75rem' }}>
          <h2>3. Route to the right Waymo roles</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
            Each card blends retrieval scores, reranker feedback, and Gemini explanations so you can take action fast.
          </p>
          <div className="match-grid">
            {matches.map((match) => {
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
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', fontSize: '0.85rem' }}>
                    <span className="tag">Retrieval {formatScore(match.retrieval_score)}</span>
                    <span className="tag">Rerank {formatScore(match.rerank_score)}</span>
                    <span className="tag">Confidence {formatScore(match.confidence)}</span>
                  </div>
                  {match.explanation && (
                    <div className="markdown-content" style={{ color: 'var(--text-muted)', lineHeight: 1.55, marginTop: '1rem' }}>
                      <ReactMarkdown>{match.explanation}</ReactMarkdown>
                    </div>
                  )}
                  {match.reason_codes && match.reason_codes.length > 0 && (
                    <div>
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
                      style={{ marginTop: 'auto' }}
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
            })}
          </div>
        </section>
      )}
    </main>
  );
}
