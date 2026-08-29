"""Entry point for `python -m ebooker`.

Without this, the obvious-looking `python -m ebooker.cli` imports the module
and exits 0 having done nothing, because cli.py defines main() but never calls
it. A no-op that reports success is worse than an error: a driver script that
retries on failure sees a clean exit, records no progress, and burns its whole
attempt budget in under a second.
"""

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())  # the package-level main() reads sys.argv itself
