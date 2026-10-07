import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from forge_resshift.models import Asset, ensure_asset
            cls.asset_type = Asset
            cls.ensure_asset = staticmethod(ensure_asset)
        except ImportError as error:
            raise AssertionError("The first-use model downloader is not implemented yet") from error

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"model-weights")

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/weights"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_download_is_verified_and_reused_without_network(self):
        digest = hashlib.sha256(b"model-weights").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            asset = self.asset_type("weights.bin", self.url, 13, digest)
            path = self.ensure_asset(asset, Path(directory))
            self.assertEqual(path.read_bytes(), b"model-weights")
            offline_asset = self.asset_type("weights.bin", "http://127.0.0.1:1", 13, digest)
            self.assertEqual(self.ensure_asset(offline_asset, Path(directory)), path)
            self.assertEqual([item.name for item in Path(directory).iterdir()], ["weights.bin"])

    def test_bad_hash_never_publishes_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = self.asset_type("weights.bin", self.url, 13, "0" * 64)
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                self.ensure_asset(asset, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_incomplete_cached_download_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "weights.bin").write_bytes(b"partial")
            asset = self.asset_type("weights.bin", self.url, 13,
                                    hashlib.sha256(b"model-weights").hexdigest())
            self.assertEqual(self.ensure_asset(asset, folder).read_bytes(), b"model-weights")


if __name__ == "__main__":
    unittest.main()
