import time
import pyaudiowpatch as pyaudio
from callcribe.audio import get_loopback_devices

pa = pyaudio.PyAudio()
devices = get_loopback_devices(pa)
pa.terminate()

for d in devices:
    print(f"opening: {d['name']}  "
          f"{int(d['defaultSampleRate'])} Hz, {d['maxInputChannels']} ch", flush=True)
    pa = pyaudio.PyAudio()
    try:
        s = pa.open(format=pyaudio.paInt16,
                    channels=max(1, int(d["maxInputChannels"])),
                    rate=int(d["defaultSampleRate"]), input=True,
                    input_device_index=d["index"], frames_per_buffer=1024)
        s.start_stream(); time.sleep(1.0); s.stop_stream(); s.close()
        print("   ok", flush=True)
    except Exception as exc:
        print(f"   FAILED: {exc}", flush=True)
    pa.terminate()
print("all devices opened")
