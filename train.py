"""Training compatibility entrypoint.

Delegates to scripts.train for the standard training workflow.
"""

from scripts.train import main


if __name__ == "__main__":
    main()
