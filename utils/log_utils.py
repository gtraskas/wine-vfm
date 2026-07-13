"""Converts the agents' ANSI-colored log lines into HTML for the web UI."""

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
RESET = "\033[0m"

MAPPER = {
    BG_BLACK + RED: "#dd0000",
    BG_BLACK + GREEN: "#00dd00",
    BG_BLACK + YELLOW: "#dddd00",
    BG_BLACK + BLUE: "#6a8dd8",
    BG_BLACK + MAGENTA: "#aa00dd",
    BG_BLACK + CYAN: "#00dddd",
    BG_BLACK + WHITE: "#87CEEB",
    BG_BLUE + WHITE: "#ff7800",
}


def reformat(message: str) -> str:
    """Replace ANSI color codes with HTML spans.

    Args:
        message: A log line possibly containing ANSI escape sequences.

    Returns:
        The line with color codes converted to styled spans.
    """
    for code, color in MAPPER.items():
        message = message.replace(code, f'<span style="color: {color}">')
    return message.replace(RESET, "</span>")
