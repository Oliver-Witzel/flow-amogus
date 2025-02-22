import queue
import threading
import time
from typing import cast
from urllib.parse import urlparse

import av
import av.container
import numpy as np
import numpy.typing as npt


class Client:
  def __init__(self, rtsp_url: str) -> None:
    self.rtsp_url = rtsp_url
    self.rtsp_ip = urlparse(rtsp_url).hostname
    self.container: av.container.Container | None = None
    self.running = False
    self.decode_thread: threading.Thread | None = None
    self.frame_queue: queue.Queue[npt.NDArray[np.uint8]] = queue.Queue(maxsize=3)
    self.last_frame: npt.NDArray[np.uint8] | None = None
    self._lock = threading.Lock()

    if not self._connect():
      print(f"Failed to connect to {self.rtsp_url}")

  def _connect(self) -> bool:
    """Attempt to establish a connection to the RTSP stream with exponential backoff."""
    base_delay = 2
    max_delay = 120
    attempt = 1
    delay = base_delay

    while True:
      try:
        print(f"Connecting (attempt {attempt})...")
        with self._lock:
          self.container = av.open(
            self.rtsp_url,
            timeout=10,
            options={
              'rtsp_transport': 'tcp',
              'stimeout': '5000000',
              'buffer_size': '10240000',
              # 'hwaccel': 'cuda',
              # 'hwaccel_device': '0',
              'threads': 'auto',
              'tune': 'zerolatency',
              'fflags': 'nobuffer',
            },
          )
          self.running = True
        print("Connection established.")

        self.decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decode_thread.start()

        start_time = time.time()
        while self.frame_queue.empty() and time.time() - start_time < 2.0:
          time.sleep(0.01)

        return True
      except Exception as e:
        print(f"Connection error: {e}")
        time.sleep(delay)
        delay = min(delay * 2, max_delay)
        attempt += 1

  def _reconnect(self) -> bool:
    """Attempt to reconnect to the stream."""
    print("Reconnecting...")
    self.running = False

    try:
      with self._lock:
        if self.container is not None:
          container = cast(av.container.InputContainer, self.container)
          container.close()
          self.container = None

      with self.frame_queue.mutex:
        self.frame_queue.queue.clear()

      if threading.current_thread() != self.decode_thread:
        if self.decode_thread is not None:
          self.decode_thread.join(timeout=2.0)
          self.decode_thread = None

      time.sleep(1)
      return self._connect()

    except Exception as e:
      print(f"Error during reconnection: {e}")
      return False

  def _decode_loop(self) -> None:
    """Continuously decode frames in a background thread."""
    while self.running:
      try:
        with self._lock:
          if self.container is None:
            continue
          container = cast(av.container.InputContainer, self.container)
          packet = next(container.demux(video=0))
          for frame in packet.decode():
            if isinstance(frame, av.VideoFrame):
              frame_array = frame.to_ndarray(format='bgr24')
              while not self.frame_queue.empty():
                try:
                  self.frame_queue.get_nowait()
                except queue.Empty:
                  break
              self.frame_queue.put(frame_array)
      except (StopIteration, av.OSError) as e:
        print(f"Stream issue: {e}, attempting reconnect...")
        self._reconnect()
      except Exception as e:
        print(f"Unexpected error: {e}")
        self._reconnect()

  def read(self) -> tuple[bool, np.ndarray]:
    """Retrieve the latest decoded frame."""
    try:
      frame = self.frame_queue.get(timeout=0.005)
      self.last_frame = frame
      return True, frame
    except queue.Empty:
      if self.last_frame is not None:
        return True, self.last_frame
      return False, np.zeros((1, 1, 3), dtype=np.uint8)

  def release(self) -> None:
    """Stop decoding thread and release resources."""
    print("Releasing resources...")
    self.running = False
    if self.decode_thread and threading.current_thread() != self.decode_thread:
      self.decode_thread.join()
    with self._lock:
      if self.container is not None:
        cast(av.container.InputContainer, self.container).close()