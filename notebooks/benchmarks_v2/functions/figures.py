from pathlib import Path
import matplotlib as mpl


def save_pdf(fig, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02, dpi=dpi)
