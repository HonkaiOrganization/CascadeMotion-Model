"""Small CLI checks for split leakage, checkpoint selection and export provenance."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from click.testing import CliRunner

from script.export import export
from script.train import check_split, digest, source_files, train


class TrainingChecks(unittest.TestCase):
    def test_cli_and_split_guard(self):
        cfg = yaml.safe_load(Path('config/paper_20261011.yaml').read_text(encoding='utf-8'))
        manifest = {subset: source_files(cfg['data'], subset) for subset in ['train', 'validation']}
        check_split(manifest)
        with self.assertRaises(ValueError):
            check_split({'train': manifest['train'], 'validation': manifest['train']})
        with tempfile.TemporaryDirectory(dir='exp') as directory:
            output = Path(directory)
            cfg.update(epochs=2, min_epochs=2, device='cpu', log_every=1, model_name=str(output / 'best.pth'))
            config = output / 'input.yaml'
            config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            runner = CliRunner()
            result = runner.invoke(train, ['--config', str(config)])
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['optimizer_steps'], 2 * 11)
            self.assertEqual(summary['checkpoint_sha256'], digest(output / 'best.pth'))
            self.assertEqual(summary['selection'], 'minimum validation cross-entropy')
            package = output / 'model.pt'
            result = runner.invoke(export, ['--checkpoint', str(output / 'best.pth'), '--output', str(package), '--config', str(config)])
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            metadata = torch.load(package, weights_only=True)['metadata']
            self.assertEqual(metadata['training']['checkpoint_sha256'], digest(output / 'best.pth'))
            self.assertEqual(metadata['training_audit_sha256']['metrics.csv'], digest(output / 'metrics.csv'))
            result = runner.invoke(train, ['--config', str(config)])
            self.assertIsInstance(result.exception, FileExistsError)


if __name__ == '__main__':
    Path('exp').mkdir(exist_ok=True)
    unittest.main()
