const trimOrNull = (value?: string) => {
  if (!value) return null;
  const trimmed = value.trim();
  return trimmed.length ? trimmed : null;
};

export const API_BASE = (() => {
  const explicit = trimOrNull(process.env.NEXT_PUBLIC_API_BASE_URL);
  if (explicit) {
    return explicit;
  }
  if (process.env.NODE_ENV === 'production') {
    return '/api';
  }
  return 'http://localhost:8000';
})();
