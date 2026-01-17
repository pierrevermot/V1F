"""
Tests for configuration management.
"""

import pytest
import tempfile
import os


class TestConfig:
    """Tests for Config class."""
    
    def test_default_init(self, default_config):
        """Test default initialization."""
        assert default_config.source is not None
        assert default_config.instrument is not None
        assert default_config.training is not None
        assert default_config.data is not None
        assert default_config.paths is not None
    
    def test_source_config_defaults(self):
        """Test SourceConfig defaults."""
        from nebraa.config import SourceConfig
        
        cfg = SourceConfig()
        assert cfg.mode == 'fourier_ps'
        assert cfg.n_files == 4096
        assert cfg.n_images_per_file == 256
    
    def test_instrument_config_defaults(self):
        """Test InstrumentConfig defaults."""
        from nebraa.config import InstrumentConfig
        
        cfg = InstrumentConfig()
        assert cfg.name == 'vlt'
        assert cfg.n_pix == 512
        assert cfg.wavelength == 4.78e-6
    
    def test_to_dict(self, default_config):
        """Test conversion to dict."""
        d = default_config.to_dict()
        
        assert isinstance(d, dict)
        assert 'source' in d
        assert 'instrument' in d
    
    def test_from_dict(self):
        """Test creation from dict."""
        from nebraa.config import Config
        
        d = {
            'source': {'mode': 'fourier', 'n_files': 100},
            'instrument': {'name': 'vlt', 'n_pix': 256},
        }
        
        cfg = Config.from_dict(d)
        
        assert cfg.source.mode == 'fourier'
        assert cfg.source.n_files == 100
        assert cfg.instrument.n_pix == 256


class TestConfigIO:
    """Tests for config file I/O."""
    
    def test_save_load_yaml(self, default_config):
        """Test YAML save/load roundtrip."""
        from nebraa.config import save_config, load_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.yaml')
            
            save_config(default_config, path)
            loaded = load_config(path)
            
            assert loaded.source.mode == default_config.source.mode
            assert loaded.instrument.name == default_config.instrument.name
    
    def test_save_load_json(self, default_config):
        """Test JSON save/load roundtrip."""
        from nebraa.config import save_config, load_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            
            save_config(default_config, path)
            loaded = load_config(path)
            
            assert loaded.source.n_files == default_config.source.n_files
    
    def test_paths_ensure_dirs(self):
        """Test directory creation."""
        from nebraa.config import PathConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PathConfig(
                scratch_dir=tmpdir,
                output_dir=os.path.join(tmpdir, 'outputs'),
            )
            
            cfg.ensure_dirs()
            
            assert os.path.isdir(cfg.images_dir)
            assert os.path.isdir(cfg.observations_dir)


class TestInstrumentPresets:
    """Tests for instrument presets."""
    
    def test_apply_preset(self):
        """Test applying instrument preset."""
        from nebraa.config import Config, apply_preset
        
        cfg = Config()
        cfg = apply_preset(cfg, 'vlt_l')
        
        assert cfg.instrument.wavelength == 4.78e-6
    
    def test_unknown_preset(self):
        """Test error on unknown preset."""
        from nebraa.config import Config, apply_preset
        
        cfg = Config()
        
        with pytest.raises(ValueError):
            apply_preset(cfg, 'unknown_telescope')
