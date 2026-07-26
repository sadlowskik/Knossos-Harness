"""`python -m harness` -- start the ACP server on stdio."""
import sys

from .acp import main

if __name__ == "__main__":
    sys.exit(main())
