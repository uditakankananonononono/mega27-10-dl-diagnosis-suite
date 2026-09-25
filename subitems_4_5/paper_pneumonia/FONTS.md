Paper font: real Times New Roman (Microsoft Core Fonts, times32.exe).
The EULA permits use but NOT redistribution, so no TTF/TFM files are committed.
Regenerate locally: ./build_fonts.sh /path/to/times32.exe (needs 7z, ttf2tfm),
then pdflatex main.tex twice. Without the fonts the build falls back to the
Nimbus clone via mathptmx (committed backup preamble: main_pdflatex_backup.tex).
