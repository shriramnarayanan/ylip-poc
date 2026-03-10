import sys

if sys.platform == "win32":
    # Suppress spurious WinError 10054 (connection forcibly closed by remote host)
    # that asyncio's ProactorEventLoop raises when the browser disconnects.
    from asyncio.proactor_events import _ProactorBasePipeTransport
    _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    def _silence_connection_lost(self, exc):
        try:
            _orig_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass

    _ProactorBasePipeTransport._call_connection_lost = _silence_connection_lost

from ui.app import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme="soft",
    )
