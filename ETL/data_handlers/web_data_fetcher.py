import requests
from typing import BinaryIO


class RemoteFileFetcher:
    """Fetches files from remote sources via HTTP/HTTPS."""

    DEFAULT_TIMEOUT = 60
    DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB

    def __init__(self, user_agent: str = "AsafZenou-Research/1.0"):
        """
        Args:
            user_agent: User agent string for HTTP requests.
        """
        self.user_agent = user_agent

    def _get_headers(self) -> dict:
        """Get HTTP headers for requests."""
        return {"User-Agent": self.user_agent}

    def fetch_stream(self, url: str, timeout: int = DEFAULT_TIMEOUT):
        """
        Fetch streaming response from remote URL.

        Args:
            url: Remote URL to fetch from.
            timeout: Request timeout in seconds.

        Returns:
            Response object with stream=True.

        Raises:
            requests.RequestException: If fetch fails.
        """
        response = requests.get(
            url, headers=self._get_headers(), stream=True, timeout=timeout
        )
        response.raise_for_status()
        return response

    def get_total_size(self, response) -> int:
        """Get total file size from response headers."""
        return int(response.headers.get("content-length", 0))

    def write_chunks_to_file(
        self, response, file_handle: BinaryIO, on_chunk_written=None
    ) -> int:
        """
        Write response chunks to file.

        Args:
            response: Response object from fetch_stream.
            file_handle: Open file handle in binary write mode.
            on_chunk_written: Optional callback(bytes_written, total_size) for progress.

        Returns:
            Total bytes written.
        """
        total_written = 0
        total_size = self.get_total_size(response)

        for chunk in response.iter_content(chunk_size=self.DEFAULT_CHUNK_SIZE):
            file_handle.write(chunk)
            total_written += len(chunk)

            if on_chunk_written:
                on_chunk_written(total_written, total_size)

        return total_written
