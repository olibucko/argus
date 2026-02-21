from datetime import time as dt_time
from unittest.mock import patch

import pytest

from security_dashboard.core import (
    calculate_optimal_grid,
    create_viewport_configs_from_cameras,
    ViewportConfig,
    AlertConfig,
    AlertManager,
)


class TestCalculateOptimalGrid:

    def test_zero_cameras(self):
        assert calculate_optimal_grid(0) == (1, 1)

    def test_single_camera(self):
        assert calculate_optimal_grid(1) == (1, 1)

    def test_two_cameras(self):
        rows, cols = calculate_optimal_grid(2)
        assert rows * cols >= 2

    def test_four_cameras(self):
        assert calculate_optimal_grid(4) == (2, 2)

    def test_nine_cameras(self):
        assert calculate_optimal_grid(9) == (3, 3)

    def test_sixteen_cameras(self):
        assert calculate_optimal_grid(16) == (4, 4)

    def test_large_count_has_enough_cells(self):
        rows, cols = calculate_optimal_grid(25)
        assert rows * cols >= 25


class TestCreateViewportConfigs:

    def test_creates_config_per_enabled_camera(self):
        cameras = [
            {"camera_name": "Cam1", "name": "Front", "enabled": True},
            {"camera_name": "Cam2", "name": "Back", "enabled": True},
            {"camera_name": "Cam3", "name": "Side", "enabled": False},
        ]
        defaults = {"sensitivity": 0.5, "min_confidence": 0.5}
        configs = create_viewport_configs_from_cameras(cameras, defaults)
        assert len(configs) == 2

    def test_empty_cameras(self):
        configs = create_viewport_configs_from_cameras([], {})
        assert configs == {}

    def test_viewport_ids_are_grid_positions(self):
        cameras = [
            {"camera_name": f"Cam{i}", "name": f"View{i}", "enabled": True}
            for i in range(4)
        ]
        defaults = {}
        configs = create_viewport_configs_from_cameras(cameras, defaults)
        assert "0,0" in configs
        assert "0,1" in configs
        assert "1,0" in configs
        assert "1,1" in configs


class TestViewportConfig:

    def test_defaults(self):
        vc = ViewportConfig()
        assert vc.camera_name == "Unnamed"
        assert vc.sensitivity == 0.5
        assert vc.min_object_size == (30, 30)


class TestAlertConfig:

    def test_curfew_within_range(self):
        config = AlertConfig(
            curfew_start=dt_time(5, 0),
            curfew_end=dt_time(17, 0),
            cooldown_period=30,
            batch_window=5,
            recipient_list=["test@example.com"],
        )
        assert config.curfew_start == dt_time(5, 0)
        assert config.curfew_end == dt_time(17, 0)
        assert config.telegram_enabled is False
        assert config.email_as_fallback is True
