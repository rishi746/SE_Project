"""Start one warmed local model and the authenticated website on loopback."""
import logging
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ['SMS_INFERENCE'] = 'local'
os.environ['DJANGO_SETTINGS_MODULE'] = 'portal.local_settings'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'website')]
log_dir = ROOT / 'outputs/local-runtime'
log_dir.mkdir(parents=True, exist_ok=True)
if sys.stdout is None:
    sys.stdout = open(log_dir / 'stdout.log', 'a', encoding='utf-8', buffering=1)
if sys.stderr is None:
    sys.stderr = open(log_dir / 'stderr.log', 'a', encoding='utf-8', buffering=1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(log_dir / 'runtime.log', encoding='utf-8'), logging.StreamHandler()])

if __name__ == '__main__':
    from filelock import FileLock, Timeout
    try:
        with FileLock(str(log_dir / 'server.lock'), timeout=0):
            import django
            django.setup()
            from django.core.management import call_command
            call_command('migrate', interactive=False, verbosity=0)
            from sms_ac.local_inference import get_engine
            get_engine()
            threading.Thread(target=call_command, args=('runserver', '127.0.0.1:8001'),
                kwargs={'use_reloader': False}, daemon=True).start()
            logging.info('Local websites ready: LLM1 http://127.0.0.1:8000 and LLM2 http://127.0.0.1:8001')
            call_command('runserver', '127.0.0.1:8000', use_reloader=False)
    except Timeout:
        logging.info('The local website is already running.')
