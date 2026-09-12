import { NextResponse } from 'next/server';
import * as cheerio from 'cheerio';

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams.get('q')?.trim();
  if (!q) return NextResponse.json({results: []});
  const searchUrl = 'https://www.google.com/search?q=' + encodeURIComponent('site:raiplay.it/programmi ' + q) + '&num=20';
  try {
    const r = await fetch(searchUrl, {headers:{'User-Agent':'Mozilla/5.0 MusicReportAI/1.0'}, cache:'no-store'});
    const html = await r.text();
    const $ = cheerio.load(html);
    const results:any[]=[];
    $('a').each((_,a)=>{
      const href=$(a).attr('href')||''; const title=$(a).text().trim();
      if(href.includes('raiplay.it/programmi/') && title && !results.some(x=>x.url===href)) results.push({title,url:href});
    });
    return NextResponse.json({results:results.slice(0,12)});
  } catch(e) { return NextResponse.json({error:'Ricerca non disponibile',results:[]},{status:502}); }
}