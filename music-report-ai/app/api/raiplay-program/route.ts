import { NextResponse } from 'next/server';

function absoluteUrl(path: string) {
  if (path.startsWith('http')) return path;
  return 'https://www.raiplay.it' + (path.startsWith('/') ? path : '/' + path);
}

export async function GET(req: Request) {
  const url = new URL(req.url).searchParams.get('url')?.trim();
  if (!url || !url.includes('raiplay.it')) return NextResponse.json({ error: 'URL RaiPlay non valido' }, { status: 400 });

  const candidates = [url, url.replace(/\/$/, '') + '.json'];
  for (const candidate of candidates) {
    try {
      const r = await fetch(absoluteUrl(candidate), {
        headers: { 'User-Agent': 'Mozilla/5.0 MusicReportAI/1.0' },
        cache: 'no-store'
      });
      if (!r.ok) continue;
      const text = await r.text();
      const contentType = r.headers.get('content-type') || '';
      if (contentType.includes('json') || text.trim().startsWith('{')) {
        const data = JSON.parse(text);
        return NextResponse.json({ source: candidate, data });
      }
      return NextResponse.json({ source: candidate, html: true });
    } catch {}
  }
  return NextResponse.json({ source: url, data: null, html: true });
}
