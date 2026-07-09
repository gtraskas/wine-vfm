"""Base class for all agents: shared colored logging."""

from __future__ import annotations

import logging


class Agent:
    """Base class providing colored, prefixed logging for subclasses.

    Attributes:
        name: Human-readable agent identifier, shown in log lines.
        color: ANSI foreground color code used for this agent's log lines.
    """

    # Foreground color codes
    RED: str = "\033[31m"
    GREEN: str = "\033[32m"
    YELLOW: str = "\033[33m"
    BLUE: str = "\033[34m"
    MAGENTA: str = "\033[35m"
    CYAN: str = "\033[36m"
    WHITE: str = "\033[37m"

    # Background color codes
    BG_BLACK: str = "\033[40m"

    # Reset code to return to default color
    RESET: str = "\033[0m"

    name: str = ""
    color: str = "\033[37m"

    def log(self, message: str) -> None:
        """Log a message prefixed with the agent's name, in its color.

        Args:
            message: Text to log.
        """
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)
