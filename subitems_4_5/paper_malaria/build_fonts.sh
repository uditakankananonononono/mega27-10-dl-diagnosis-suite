#!/bin/bash
# Build Times New Roman metrics for the paper from the Microsoft Core Fonts
# installer (EULA permits use, not redistribution - TTFs and derived metrics
# stay local, only this script and the PDF are committed).
# Usage: ./build_fonts.sh /path/to/times32.exe
set -e
EXE="$1"
[ -f "$EXE" ] || { echo "need times32.exe (see paper/README)"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP=$(mktemp -d)
7z e -y -o"$TMP" "$EXE" '*.TTF' >/dev/null
ENC=$(kpsewhich ec.enc)
for pair in "Times.TTF tnr-ec" "Timesbd.TTF tnrbd-ec" "Timesi.TTF tnri-ec" "Timesbi.TTF tnrbi-ec"; do
  set -- $pair
  ttf2tfm "$TMP/$1" -q -p "$ENC" "$HERE/$2"
  cp "$TMP/$1" "$HERE/$1"
done
cat > "$HERE/tnr-ec.map" <<MAP
tnr-ec TimesNewRomanPSMT " ecEncoding ReEncodeFont " <$ENC <Times.TTF
tnrbd-ec TimesNewRomanPS-BoldMT " ecEncoding ReEncodeFont " <$ENC <Timesbd.TTF
tnri-ec TimesNewRomanPS-ItalicMT " ecEncoding ReEncodeFont " <$ENC <Timesi.TTF
tnrbi-ec TimesNewRomanPS-BoldItalicMT " ecEncoding ReEncodeFont " <$ENC <Timesbi.TTF
MAP
rm -rf "$TMP"
echo "fonts built in $HERE - rebuild the PDF with pdflatex"
