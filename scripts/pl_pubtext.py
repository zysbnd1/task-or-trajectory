# SPDX-License-Identifier: Apache-2.0
"""paper-lab: print the publishable text of a writeup draft (all HTML comments removed).
Word count as the lead measures it:  python3 scripts/pl_pubtext.py WRITEUP_draft_v0.md | wc -w   (target <= 2,800)."""
import re
import sys

print(re.sub(r"<!--.*?-->\n?", "", open(sys.argv[1]).read(), flags=re.S).strip())
