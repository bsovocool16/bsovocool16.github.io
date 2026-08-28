#!/bin/bash
# House style: less-assimilated Latin phrases are italicized.
# Fully-assimilated legal Latin (per se, de facto, prima facie, res judicata,
# de novo, en banc, e.g., i.e.) stays roman, per usual legal-writing practice.
# Usage: tools/latin-check.sh   (run from repo root before pushing)

PHRASES="ex ante|ex post|in situ|in limine|ejusdem generis|sui generis|inter alia|mutatis mutandis|ceteris paribus|a fortiori|ipso facto|sub silentio|ultra vires|arguendo|in pari materia|expressio unius|noscitur a sociis|contra proferentem|ex parte|obiter dictum|polis"

found=0
for f in notes/*.html index.html; do
  [ -e "$f" ] || continue
  # strip italicized instances, then look for what remains
  if hits=$(perl -pe 's{<i>(.*?)</i>}{}gs' "$f" | grep -o -i -E "$PHRASES" | sort -u); then
    if [ -n "$hits" ]; then
      echo "$f:"
      echo "$hits" | sed 's/^/  not italicized: /'
      found=1
    fi
  fi
done
[ "$found" -eq 0 ] && echo "OK: all Latin phrases italicized"
exit 0
