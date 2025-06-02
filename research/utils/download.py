from pathlib import Path

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def download_url(url, file_path: str | Path):
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        task = progress.add_task(f"Download from {url}", total=total_size)
        with Path(file_path).open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress.update(task, advance=size)


def download_google(file_id, file_path):
    download_url(
        f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t",
        file_path,
    )
