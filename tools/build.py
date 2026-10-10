#!/usr/bin/env python3
"""Bygger webappen ud fra prototypen (artefaktets HTML).

    python3 tools/build.py sti/til/prototype.html

Hovedet fra den nuværende index.html (lokale skrifttyper, manifest, ikoner) og
webapp-stilene bevares. Prototypens indhold sættes ind, APP slås til, koden til
automatisk opdatering og nyhedsvinduet (changelog.json) sættes ind, og
cachenavnet i sw.js skiftes, så telefoner henter den nye udgave.
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX, SW = ROOT / 'index.html', ROOT / 'sw.js'
CHANGELOG, NEWS = ROOT / 'changelog.json', ROOT / 'tools' / 'changelog.html'

REGISTER = "  if ('serviceWorker' in navigator) window.addEventListener('load', () => { navigator.serviceWorker.register('sw.js').catch(() => {}); });\n"
AUTO_UPDATE = """  /* Ny udgave: når appen åbnes eller kommer frem igen, ser den efter en ny sw.js. Findes der en, overtager den, og siden genindlæses én gang med de nye filer. */
  if ('serviceWorker' in navigator) {
    const hadSW = !!navigator.serviceWorker.controller; let reloaded = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => { if (hadSW && !reloaded) { reloaded = true; persist(); location.reload(); } });
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('sw.js', { updateViaCache: 'none' }).then(reg => {
        reg.update().catch(() => {});
        document.addEventListener('visibilitychange', () => { if (!document.hidden) reg.update().catch(() => {}); });
      }).catch(() => {});
    });
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
    styles = cut(proto, '<style>\n/* Layout:', '<div class="wrap">', 'prototypens stilark')
    body = proto[proto.index('<div class="wrap">'):]
    body = re.sub(r'\s*</body>\s*</html>\s*$', '\n', body)

    if '/*APPMODE*/false' not in body:
        sys.exit('Fandt ikke /*APPMODE*/false i prototypen')
    body = body.replace('/*APPMODE*/false', 'true')
    if REGISTER not in body:
        sys.exit('Fandt ikke registreringen af sw.js i prototypen')
    body = body.replace(REGISTER, AUTO_UPDATE)

    log = json.loads(CHANGELOG.read_text(encoding='utf-8'))
    if [e['version'] for e in log] != sorted((e['version'] for e in log), reverse=True):
        sys.exit('changelog.json skal stå med den nyeste udgave først')
    news = NEWS.read_text(encoding='utf-8').replace('/*CHANGELOG*/[]', json.dumps(log, ensure_ascii=False).replace('</', '<\\/'))

    out = head + styles.rstrip() + '\n\n' + webapp + '</head>\n<body class="app">\n' + body + '\n' + news + '</body>\n</html>\n'
    INDEX.write_text(out, encoding='utf-8')

    version = hashlib.sha1(out.encode('utf-8')).hexdigest()[:10]
    sw = SW.read_text(encoding='utf-8')
    SW.write_text(re.sub(r'hiero-proto-[0-9a-f]+', 'hiero-proto-' + version, sw), encoding='utf-8')
    print('index.html bygget, cache hiero-proto-' + version)


if __name__ == '__main__':
    main()
