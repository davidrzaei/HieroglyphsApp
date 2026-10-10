#!/usr/bin/env python3
"""Bygger webappen ud fra prototypen (artefaktets HTML).

    python3 tools/build.py sti/til/prototype.html

Hovedet fra den nuværende index.html (lokale skrifttyper, manifest, ikoner) og
webapp-stilene bevares. Prototypens indhold sættes ind, APP slås til, koden til
automatisk opdatering, nyhedsvinduet (changelog.json) og knappen
"Check for updates" under Settings sættes ind, og
cachenavnet i sw.js skiftes, så telefoner henter den nye udgave.
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX, SW = ROOT / 'index.html', ROOT / 'sw.js'
CHANGELOG, NEWS, UPDATE = ROOT / 'changelog.json', ROOT / 'tools' / 'changelog.html', ROOT / 'tools' / 'update.html'

REGISTER = "  if ('serviceWorker' in navigator) window.addEventListener('load', () => { navigator.serviceWorker.register('sw.js').catch(() => {}); });\n"
AUTO_UPDATE = """  /* Ny udgave: når appen åbnes eller kommer frem igen, ser den efter en ny sw.js. Findes der en, overtager den, og siden genindlæses én gang med de nye filer. */
  if ('serviceWorker' in navigator) {
    const hadSW = !!navigator.serviceWorker.controller; let reloaded = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => { if (hadSW && !reloaded) { reloaded = true; persist(); location.reload(); } });
    const register = () => navigator.serviceWorker.register('sw.js', { updateViaCache: 'none' }).then(reg => {
      reg.update().catch(() => {});
      document.addEventListener('visibilitychange', () => { if (!document.hidden) reg.update().catch(() => {}); });
    }).catch(() => {});
    /* Koden kører først, når databasen er åben, og da er siden ofte allerede indlæst. */
    if (document.readyState === 'complete') register(); else window.addEventListener('load', register);
  }
"""


def cut(text, start, end, what):
    i = text.find(start)
    j = text.find(end, i)
    if i < 0 or j < 0:
        sys.exit(f'Fandt ikke {what}')
    return text[i:j]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    proto = Path(sys.argv[1]).read_text(encoding='utf-8')
    cur = INDEX.read_text(encoding='utf-8')

    head = cur[:cur.index('<title>Hieroglyphs</title>') + len('<title>Hieroglyphs</title>\n')]
    webapp = cut(cur, '<style>\n/* Webapp:', '</head>', 'webapp-stilene i index.html')
    fonts = proto.find('fonts.googleapis.com')
    styles = cut(proto[fonts:] if fonts >= 0 else proto, '<style>\n', '<div class="wrap">', 'prototypens stilark')
    body = proto[proto.index('<div class="wrap">'):]
    body = re.sub(r'\s*</body>\s*</html>\s*$', '\n', body)

    if '/*APPMODE*/false' not in body:
        sys.exit('Fandt ikke /*APPMODE*/false i prototypen')
    body = body.replace('/*APPMODE*/false', 'true')
    if REGISTER not in body:
        sys.exit('Fandt ikke registreringen af sw.js i prototypen')
    body = body.replace(REGISTER, AUTO_UPDATE)
    # Webappen bruger sin egen kopi af SQLite først, så den ikke afhænger af et CDN og virker uden net.
    body = re.sub(r'const SQLITE_URLS = \[("https://[^"]+"), ("vendor/[^"]+")\];', r'const SQLITE_URLS = [\2, \1];', body)

    log = json.loads(CHANGELOG.read_text(encoding='utf-8'))
    if [e['version'] for e in log] != sorted((e['version'] for e in log), reverse=True):
        sys.exit('changelog.json skal stå med den nyeste udgave først')
    news = NEWS.read_text(encoding='utf-8').replace('/*CHANGELOG*/[]', json.dumps(log, ensure_ascii=False).replace('</', '<\\/'))
    news += UPDATE.read_text(encoding='utf-8').replace("/*VERSION*/''", json.dumps(log[0]['version'] if log else ''))

    out = head + styles.rstrip() + '\n\n' + webapp + '</head>\n<body class="app">\n' + body + '\n' + news + '</body>\n</html>\n'
    INDEX.write_text(out, encoding='utf-8')

    version = hashlib.sha1(out.encode('utf-8')).hexdigest()[:10]
    # Filerne, appen gemmer til brug uden net. Databasen (data/) gemmer appen selv i IndexedDB.
    files = ['./', 'index.html', 'manifest.webmanifest'] + sorted(
        f.relative_to(ROOT).as_posix() for d in ('icons', 'fonts', 'vendor') for f in (ROOT / d).rglob('*')
        if f.is_file() and f.suffix in ('.png', '.woff2', '.ttf', '.mjs', '.wasm'))
    sw = SW.read_text(encoding='utf-8')
    sw = re.sub(r'hiero-proto-[0-9a-f]+', 'hiero-proto-' + version, sw)
    sw = re.sub(r'const FILES = \[.*?\];', lambda m: 'const FILES = [ ' + ', '.join(json.dumps(f) for f in files) + ' ];', sw, flags=re.S)
    SW.write_text(sw, encoding='utf-8')
    print('index.html bygget, cache hiero-proto-' + version)


if __name__ == '__main__':
    main()
