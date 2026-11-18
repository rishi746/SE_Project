import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/kaggle_job.py'
spec = importlib.util.spec_from_file_location('kaggle_job', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackageTests(unittest.TestCase):
    def test_package_uses_allowlist_and_excludes_credentials_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['src/sms_ac/pipeline.py', 'src/sms_ac/__pycache__/x.pyc',
                         'configs/kaggle.json', 'configs/access_token', 'configs/.env',
                         '.venv/private.txt', 'personal.txt', 'README.md', 'data/dual/design/train.jsonl']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture', encoding='utf-8')
            included = {p.relative_to(root).as_posix() for p in module.package_files(root)}
            self.assertEqual(included, {'src/sms_ac/pipeline.py', 'README.md', 'data/dual/design/train.jsonl'})

    def test_username_is_not_shell_code(self):
        for username in ['https://kaggle.com/a', 'a b', '../escape', '$(cmd)', 'a@b.com']:
            with self.assertRaises(ValueError):
                module.check_username(username)


if __name__ == '__main__':
    unittest.main()
