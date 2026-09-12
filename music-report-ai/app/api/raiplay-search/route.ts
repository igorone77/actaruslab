import { NextResponse } from 'next/server';
import * as cheerio from 'cheerio';

function normalize(s: string) {
  return s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').trim();
}

async function googleSearch(q: string) {
  const searchUrl = 'https://www.google.com/search?q=' + encodeURIComponent('site:raiplay.it/programmi ' + q) + '&num=20';
  const r = await fetch(searchUrl, { headers: { 'User-Agent': 'Mozilla/5.0 MusicReportAI/1.0' }, cache: 'no-store' });
  const html = await r.text();
  const $ = cheerio.load(html);
  const results: any[] = [];
  $('a').each((_, a) => {
    const href = $(a).attr('href') || '';
    const title = $(a).text().trim();
    if (href.includes('raiplay.it/programmi/') && title && !results.some(x => x.url === href)) {
      results.push({ title, url: href });
    }
  });
  return results.slice(0, 20);
}

async function raiProgramIndex() {
  const url = 'https://www.raiplay.it/dl/RaiTV/RaiPlayMobile/Prod/Config/programmiAZ-elenco.json';
  const r = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0 MusicReportAI/1.0' }, cache: 'no-store' });
  if (!r.ok) return [];
  const data = await r.json();
  const out: any[] = [];
  const walk = (x: any) => {
    if (!x) return;
    if (Array.isArray(x)) return x.forEach(walk);
    if (typeof x !== 'object') return;
    const name = x.name || x.title || x.nome;
    const path = x.pathID || x.path || x.url || x.link;
    if (name && path && String(path).includes('/programmi/')) out.push({ title: String(name), url: String(path) });
    Object.values(x).forEach(walk);
  };
  walk(data);
  return out;
}

export async function GET(req: Request) {
  const params = new URL(req.url).searchParams;
  const q = params.get('q')?.trim();
  const date = params.get('date')?.trim();
  if (!q) return NextResponse.json({ results: [] });
  const nq = normalize(q);
  const results: any[] = [];

  try {
    const indexed = await raiProgramIndex();
    for (const item of indexed) {
      if (normalize(item.title).includes(nq) || nq.includes(normalize(item.title))) {
        if (!results.some(x => x.url === item.url)) results.push(item);
      }
    }
  } catch {}

  try {
    const google = await googleSearch(q + (date ? ' ' + date : ''));
    for (const item of google) if (!results.some(x => x.url === item.url)) results.push(item);
  } catch {}

  return NextResponse.json({
    results: results.slice(0, 20),
    query: q,
    date: date || null,
    source: 'RaiPlay public catalogue + web index'
  });
}
